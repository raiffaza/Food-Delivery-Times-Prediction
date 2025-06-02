import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# Function to load files directly from GitHub
def load_file_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        print("File downloaded successfully")
        return joblib.load(BytesIO(response.content))
    else:
        print("Error downloading the file")
        return None

# URLs for model and scaler files hosted on GitHub
model_url = "https://github.com/raiffaza/Food-Delivery-Times-Prediction/raw/main/xgb_tuned_model.pkl" 
scaler_url = "https://github.com/raiffaza/Food-Delivery-Times-Prediction/raw/main/scaler.pkl" 

# Load model and scaler from GitHub
model = load_file_from_github(model_url)
scaler = load_file_from_github(scaler_url)

if not model or not scaler:
    st.error("Failed to load the model or scaler. Please check your internet connection.")
    st.stop()

# Get the exact feature names and order expected by the model
expected_columns = model.feature_names_in_.tolist()

# Identify numeric features (must match training data)
numeric_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

def make_prediction(distance_km, prep_time, courier_exp, weather, traffic, time_of_day, vehicle):
    # Initialize all features to 0
    data = {col: 0 for col in expected_columns}

    # Fill numeric features
    data['Distance_km'] = distance_km
    data['Preparation_Time_min'] = prep_time
    data['Courier_Experience_yrs'] = courier_exp

    # Set selected categorical features to 1
    weather_col = f'Weather_{weather}'
    if weather_col in data:
        data[weather_col] = 1

    traffic_col = f'Traffic_Level_{traffic}'
    if traffic_col in data:
        data[traffic_col] = 1

    time_col = f'Time_of_Day_{time_of_day}'
    if time_col in data:
        data[time_col] = 1

    vehicle_col = f'Vehicle_Type_{vehicle}'
    if vehicle_col in data:
        data[vehicle_col] = 1

    # Convert to DataFrame with the expected columns
    input_df = pd.DataFrame([data], columns=expected_columns)

    # Scale only numeric features
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # Make prediction
    prediction = model.predict(input_df)[0]
    return prediction

# Set page config
st.set_page_config(
    page_title="Uber Eats Delivery Time Prediction",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and styling
st.markdown(
    """
    <style>
    body {
        color: white;
        background-color: #1e1e1e;
    }
    .st-bf {
        color: white;
    }
    .st-ag {
        background-color: #262730;
    }
    .st-at {
        background-color: #31333F;
    }
    .st-ak {
        background-color: #31333F;
    }
    .st-al {
        background-color: #31333F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.title("Uber Eats Delivery Time Prediction")
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Uber_Eats_Logo.svg/1200px-Uber_Eats_Logo.svg.png'  
             alt='Uber Eats Logo' width='200'>
    </div>
    """,
    unsafe_allow_html=True,
)
st.subheader("Fast • Secure • Intelligent")

# About Uber Eats Section
st.header("About Uber Eats")
st.markdown(
    """
    Uber Eats is one of the leading food delivery platforms globally, serving millions of customers. By leveraging cutting-edge machine learning techniques, Uber Eats aims to provide the most accurate delivery time estimates to ensure a seamless customer experience.
    
    This application uses a trained machine learning model to predict delivery times based on various input parameters such as distance, weather, traffic, and more.
    """
)

# Business Problem Section
st.header("Business Problem")
st.markdown(
    """
    Predicting delivery times is a critical component of efficient operations at Uber Eats. Accurate delivery time predictions:
    - Improve customer satisfaction by reducing delays.
    - Optimize resource allocation for couriers and delivery routes.
    - Help manage customer expectations and improve the overall delivery process.
    
    Accurate predictions rely on understanding various factors such as traffic, weather, and the courier’s experience, making machine learning an ideal solution for the task.
    """
)
st.image("uber eats business problem.jpeg", caption="Understanding the Business Problem", use_column_width=True)

# Purpose of the Website Section
st.header("Purpose of This Tool")
st.markdown(
    """
    This interactive platform allows users to input specific delivery parameters and instantly receive a data-driven estimate of the delivery time, powered by an advanced XGBoost machine learning model.
    
    - **Real-Time Predictions**: Get instant predictions on delivery times based on distance, weather, and traffic conditions.
    - **Improved Operations**: This model helps Uber Eats enhance its operations by providing more accurate predictions for better resource allocation.
    """
)
st.image("uber eats company profile.jpeg", caption="Enhancing Uber Eats Operations", use_column_width=True)

# Input Form Section
st.header("Enter Delivery Details")
with st.form("delivery_form"):
    # Numeric Inputs
    distance_km = st.number_input("Distance (km)", min_value=0.0, format="%.2f", help="Distance between restaurant and delivery address.")
    prep_time = st.number_input("Preparation Time (minutes)", min_value=0, help="Time taken to prepare the order.")
    courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, format="%.1f", help="Years of delivery courier experience.")

    # Categorical Inputs
    weather = st.selectbox("Weather Condition", ['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'])
    traffic = st.selectbox("Traffic Level", ['Low', 'Medium', 'High'])
    time_of_day = st.selectbox("Time of Day", ['Afternoon', 'Evening', 'Night', 'Morning'])
    vehicle = st.selectbox("Vehicle Type", ['Scooter', 'Bike', 'Car'])

    # Submit Button
    submit = st.form_submit_button("Predict Delivery Time")

# Prediction Result Section
if submit:
    # Call prediction function when form is submitted
    predicted_time = make_prediction(distance_km, prep_time, courier_exp, weather, traffic, time_of_day, vehicle)
    
    # Display result in an attractive manner
    st.markdown("<h2 style='color: white; text-align: center;'>📊 Prediction Result</h2>", unsafe_allow_html=True)
    st.success(f"✅ Estimated Delivery Time: {predicted_time:.2f} minutes", icon="💨")

    # Additional Info (Explanation) centered
    st.markdown("""
    ---
    ### 🔍 Explanation of Features

    - `Distance (km)` (numeric): Distance between the restaurant and the delivery address.
    - `Preparation Time (minutes)` (numeric): Time taken to prepare the order.
    - `Courier Experience (years)` (numeric): Years of experience of the courier.
    - `Weather Condition` (categorical): Weather during the delivery (Windy, Clear, etc.).
    - `Traffic Level` (categorical): Traffic conditions during the delivery (Low, Medium, High).
    - `Time of Day` (categorical): Time of day during the delivery (Morning, Afternoon, Evening, Night).
    - `Vehicle Type` (categorical): Type of vehicle used by the courier (Scooter, Bike, Car).
    """, unsafe_allow_html=True)
