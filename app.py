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
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return joblib.load(BytesIO(response.content))
        st.error(f"❌ Failed to download model from {url}")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

def make_prediction(model, expected_columns, numeric_features, input_data):
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
            st.warning(f"⚠️ Column '{col_name}' not found in model features.")
    
    # Ensure the data matches the model columns
    data = {col: data.get(col, 0) for col in expected_columns}
    input_df = pd.DataFrame([data], columns=expected_columns)
    
    try:
        prediction = model.predict(input_df)[0]
        return prediction
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Uber Eats Delivery Time Prediction",
        page_icon="🍔",
        layout="centered"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("uber eats.png", width=250)

    st.title("Uber Eats Delivery Time Prediction")
    st.markdown("""
    ### Uber Eats is revolutionizing food delivery by leveraging cutting-edge machine learning techniques.
    Our goal is to provide the most accurate delivery time estimates to ensure a seamless customer experience.
    This app uses a trained machine learning model to predict the delivery time based on input parameters such as distance, weather, traffic, and more.
    """)

    model = load_file_from_github(MODEL_URL)
    if model is None:
        return

    expected_columns = model.feature_names_in_.tolist()
    numeric_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

    st.markdown("### Enter Delivery Details")

    with st.form("delivery_form"):
        distance_km = st.number_input("Distance (km)", min_value=0.0, format="%.2f")
        prep_time = st.number_input("Preparation Time (minutes)", min_value=0)
        courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, format="%.1f")

        weather = st.selectbox("Weather Condition", ['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'])
        traffic = st.selectbox("Traffic Level", ['Low', 'Medium', 'High'])
        time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
        vehicle = st.selectbox("Vehicle Type", ['Scooter', 'Bike', 'Car'])

        submit = st.form_submit_button("Predict Delivery Time")

    if submit:
        if distance_km == 0 or prep_time == 0:
            st.error("🚫 Distance and Preparation Time must be greater than 0.")
            return

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
        if predicted_time is not None:
            st.markdown("<h2 style='text-align: center;'>📊 Prediction Result</h2>", unsafe_allow_html=True)
            st.success(f"✅ Estimated Delivery Time: {predicted_time:.2f} minutes", icon="💨")

            st.markdown("""
            ---
            ### 🔍 Feature Guide

            - **Distance (km)**: Distance between the restaurant and the delivery address.
            - **Preparation Time**: Time required to prepare the order.
            - **Courier Experience**: Years of courier delivery experience.
            - **Weather, Traffic, Time, Vehicle**: Contextual factors influencing delivery time.
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
