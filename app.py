import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# --- Constants ---
MODEL_URL = "https://github.com/raiffaza/Food-Delivery-Times-Prediction/raw/main/xgb_tuned_model.pkl" 

# --- Helper Functions ---
@st.cache_resource
def load_file_from_github(url):
    """Load a file from GitHub using requests and joblib."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return joblib.load(BytesIO(response.content))
        st.error(f"Failed to download file from {url}")
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def make_prediction(model, expected_columns, numeric_features, input_data):
    """Make a prediction using the model without scaling."""
    # Initialize all features to 0
    data = {col: 0 for col in expected_columns}
    
    # Fill numeric features
    for key in numeric_features:
        if key in input_data:
            data[key] = input_data[key]
    
    # Fill categorical features (one-hot encoding)
    for cat in ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']:
        col_name = f"{cat}_{input_data[cat]}"
        if col_name in data:
            data[col_name] = 1
        else:
            print(f"⚠️ Warning: Column '{col_name}' not found in model features.")
    
    # Ensure the data dictionary matches the expected columns
    data = {col: data.get(col, 0) for col in expected_columns}
    
    # Convert to DataFrame with the correct columns
    input_df = pd.DataFrame([data], columns=expected_columns)

    # Make the prediction
    prediction = model.predict(input_df)[0]
    return prediction

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Uber Eats Delivery Time Prediction",
        page_icon="🍔",
        layout="centered"  # Centered layout for all content
    )

    # --- Header Section (Centered Logo) ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("uber eats.png", width=250)  # Uber Eats logo centered

    # --- Title and Description Section ---
    st.title("Uber Eats Delivery Time Prediction")
    st.markdown("""
    ### Uber Eats is revolutionizing food delivery by leveraging cutting-edge machine learning techniques.
    Our goal is to provide the most accurate delivery time estimates to ensure a seamless customer experience.
    This app uses a trained machine learning model to predict the delivery time based on input parameters such as distance, weather, traffic, and more.
    """)

    # --- Load Model ---
    model = load_file_from_github(MODEL_URL)
    if model is None:
        st.error("Failed to load model. Please check the URL or try again later.")
        return

    # Ensure expected_columns is loaded from the model
    expected_columns = model.feature_names_in_.tolist()  # Ensure this comes from the model
    numeric_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

    # --- Display Images and Text About Website Professionalism ---
    st.markdown("<h2 style='text-align: center;'>Website Overview</h2>", unsafe_allow_html=True)
    st.markdown("""
    The Streamlit web application has been designed to look professional and user-friendly, ensuring an intuitive and engaging user experience.
    Below are some visual elements that highlight the app's design and functionality:
    """)

    # Display images
    image_paths = [
        "uber eats business problem.jpeg",
        "uber eats company profile.jpeg",
        "uber eats purpose.jpg"
    ]
    for path in image_paths:
        st.image(path, caption=path, use_column_width=True)

    # Add text about professionalism
    st.markdown("""
    - **Professional Design:** The app features a clean and modern interface, making it easy to navigate.
    - **User-Friendly Inputs:** All necessary fields are clearly labeled and organized for smooth interaction.
    - **Real-Time Predictions:** Users receive instant feedback after submitting inputs, enhancing usability.
    - **Educational Insights:** Detailed explanations of each feature help users understand how predictions are made.

    Overall, the Streamlit app provides a polished and reliable platform for predicting delivery times, aligning perfectly with Uber Eats' commitment to excellence.
    """)

    # --- Input Form Section ---
    st.markdown("### Enter Delivery Details", unsafe_allow_html=True)

    with st.form("delivery_form"):
        distance_km = st.number_input("Distance (km)", min_value=0.0, format="%.2f", help="Distance between restaurant and delivery address.")
        prep_time = st.number_input("Preparation Time (minutes)", min_value=0, help="Time taken to prepare the order.")
        courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, format="%.1f", help="Years of delivery courier experience.")

        st.markdown("<span style='color:white; font-weight:bold;'>Weather Condition</span>", unsafe_allow_html=True)
        weather = st.selectbox("", ['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'])

        st.markdown("<span style='color:white; font-weight:bold;'>Traffic Level</span>", unsafe_allow_html=True)
        traffic = st.selectbox("", ['Low', 'Medium', 'High'])

        st.markdown("<span style='color:white; font-weight:bold;'>Time of Day</span>", unsafe_allow_html=True)
        time_of_day = st.selectbox("", ['Afternoon', 'Evening', 'Night', 'Morning'])

        st.markdown("<span style='color:white; font-weight:bold;'>Vehicle Type</span>", unsafe_allow_html=True)
        vehicle = st.selectbox("", ['Scooter', 'Bike', 'Car'])

        submit = st.form_submit_button("Predict Delivery Time")

    # --- PREDICTION RESULT SECTION ---
    if submit:
        input_data = {
            'Distance_km': distance_km,
            'Preparation_Time_min': prep_time,
            'Courier_Experience_yrs': courier_exp,
            'Weather': weather,
            'Traffic_Level': traffic,
            'Time_of_Day': time_of_day,
            'Vehicle_Type': vehicle
        }
        predicted_time = make_prediction(model, expected_columns, numeric_features, input_data)
        st.markdown("<h2 style='color: white; text-align: center;'>📊 Prediction Result</h2>", unsafe_allow_html=True)
        st.success(f"✅ Estimated Delivery Time: {predicted_time:.2f} minutes", icon="💨")

        st.markdown("""
        ---
        ### 🔍 Explanation of Features

        - `Distance (km)`: Distance between the restaurant and the delivery address.
        - `Preparation Time (minutes)`: Time taken to prepare the order.
        - `Courier Experience (years)`: Years of experience of the courier.
        - `Weather Condition`: Weather during the delivery (Windy, Clear, etc.).
        - `Traffic Level`: Traffic conditions during the delivery (Low, Medium, High).
        - `Time of Day`: Time of day during the delivery (Morning, Afternoon, Evening, Night).
        - `Vehicle Type`: Type of vehicle used by the courier (Scooter, Bike, Car).
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
