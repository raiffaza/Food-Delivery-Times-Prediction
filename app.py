import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# --- Constants and Config ---
MODEL_URL = "https://github.com/raiffaza/Food-Delivery-Times-Prediction/raw/main/xgb_tuned_model.pkl"
SCALER_URL = "https://github.com/raiffaza/Food-Delivery-Times-Prediction/raw/main/scaler.pkl"

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

def make_prediction(model, scaler, expected_columns, numeric_features, input_data):
    """Make a prediction using the model and scaler."""
    data = {col: 0 for col in expected_columns}
    for key, value in input_data.items():
        if key in expected_columns:
            data[key] = value
        elif key.startswith(('Weather_', 'Traffic_Level_', 'Time_of_Day_', 'Vehicle_Type_')):
            data[key] = 1 if key in expected_columns else 0

    input_df = pd.DataFrame([data], columns=expected_columns)
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    prediction = model.predict(input_df)[0]
    return prediction

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Uber Eats Delivery Time Prediction",
        page_icon="🍔",
        layout="wide"
    )

    # Load model and scaler (cached)
    model = load_file_from_github(MODEL_URL)
    scaler = load_file_from_github(SCALER_URL)

    if model is None or scaler is None:
        st.error("Failed to load model or scaler. Please check the URLs or try again later.")
        return

    # Get expected feature names and order
    expected_columns = model.feature_names_in_.tolist()
    numeric_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

    # --- UI: Logo and Title ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("uber eats.png", width=250)  # Ensure this image is in the same directory

    st.markdown("<h1 style='color: white; text-align: center;'>Uber Eats Delivery Time Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### Uber Eats is revolutionizing food delivery by leveraging cutting-edge machine learning techniques.
    Our goal is to provide the most accurate delivery time estimates to ensure a seamless customer experience.
    This app uses a trained machine learning model to predict the delivery time based on input parameters such as distance, weather, traffic, and more.
    """, unsafe_allow_html=True)

    # --- Business Problem ---
    st.markdown("""
    ## Business Problem
    Predicting delivery times is a crucial component of efficient operations at Uber Eats. Accurate delivery time predictions:
    - Improve customer satisfaction by reducing delays.
    - Optimize resource allocation for couriers and delivery routes.
    - Help manage customer expectations and improve the overall delivery process.
    """, unsafe_allow_html=True)
    st.image("uber eats business problem.jpeg", use_column_width=True)

    # --- Purpose of the Website ---
    st.markdown("""
    ## Purpose of this Website
    This platform allows users to input specific delivery parameters and instantly receive a **data-driven estimate** of the delivery time, powered by an advanced **XGBoost machine learning model**.
    - **Real-Time Predictions**: Get instant predictions on delivery times based on distance, weather, and traffic conditions.
    - **Improved Operations**: This model helps Uber Eats enhance its operations by providing more accurate predictions for better resource allocation.
    """, unsafe_allow_html=True)
    st.image("uber eats company profile.jpeg", use_column_width=True)

    # --- Input Section ---
    st.header("Enter Delivery Details", help="Please fill in the following details to estimate delivery time.")
    with st.form("delivery_form"):
        distance_km = st.number_input("Distance (km)", min_value=0.0, format="%.2f", help="Distance between restaurant and delivery address.")
        prep_time = st.number_input("Preparation Time (minutes)", min_value=0, help="Time taken to prepare the order.")
        courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, format="%.1f", help="Years of delivery courier experience.")

        weather = st.selectbox("Weather Condition", ['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'])
        traffic = st.selectbox("Traffic Level", ['Low', 'Medium', 'High'])
        time_of_day = st.selectbox("Time of Day", ['Afternoon', 'Evening', 'Night', 'Morning'])
        vehicle = st.selectbox("Vehicle Type", ['Scooter', 'Bike', 'Car'])

        submitted = st.form_submit_button("Predict Delivery Time")

    # --- Prediction and Explanation ---
    if submitted:
        # Prepare input data
        input_data = {
            'Distance_km': distance_km,
            'Preparation_Time_min': prep_time,
            'Courier_Experience_yrs': courier_exp,
            f'Weather_{weather}': 1,
            f'Traffic_Level_{traffic}': 1,
            f'Time_of_Day_{time_of_day}': 1,
            f'Vehicle_Type_{vehicle}': 1
        }

        # Make prediction
        predicted_time = make_prediction(model, scaler, expected_columns, numeric_features, input_data)
        
        # Display result
        st.markdown("<h2 style='color: white; text-align: center;'>📊 Prediction Result</h2>", unsafe_allow_html=True)
        st.success(f"✅ Estimated Delivery Time: {predicted_time:.2f} minutes", icon="💨")

        # Explanation
        st.markdown("""
        ---
        ### 🔍 Explanation of Features

        - **Distance (km)**: Distance between the restaurant and the delivery address.
        - **Preparation Time (minutes)**: Time taken to prepare the order.
        - **Courier Experience (years)**: Years of experience of the courier.
        - **Weather Condition**: Weather during the delivery (Windy, Clear, etc.).
        - **Traffic Level**: Traffic conditions during the delivery (Low, Medium, High).
        - **Time of Day**: Time of day during the delivery (Morning, Afternoon, Evening, Night).
        - **Vehicle Type**: Type of vehicle used by the courier (Scooter, Bike, Car).
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
