import streamlit as st
import pandas as pd
import joblib

# Inject custom CSS for styling
def local_css():
    st.markdown(
        """
        <style>
        /* General styles */
        body {
            background-color: #121212;
            color: #f0f0f0;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #000000;
            color: #ffffff;
        }
        h1, h2, h3, h4, h5 {
            color: #ffffff;
            font-weight: 700;
        }
        .stButton>button {
            background-color: #1DB954;
            color: white;
            font-weight: 700;
            border-radius: 8px;
            height: 45px;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1ed760;
            color: #000;
        }
        .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div {
            background-color: #222222;
            color: #f0f0f0;
            border: none;
            border-radius: 6px;
            padding: 10px;
        }
        .css-1kyxreq.edgvbvh3 {  /* selectbox arrow */
            color: #1DB954;
        }
        .stAlert {
            background-color: #1DB954 !important;
            color: #000000 !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# Load model and scaler
model = joblib.load('xgb_tuned_model.pkl')
scaler = joblib.load('scaler.pkl')

# Sidebar navigation
st.sidebar.title("Uber Delivery Insights")
menu = st.sidebar.radio("Navigate", ["Company Profile", "Business Problem", "Purpose of this Website", "Predict Delivery Time"])

# Sidebar content
if menu == "Company Profile":
    st.sidebar.markdown("""
    **Uber Technologies Inc.**  
    A global leader in transportation and delivery technology, committed to providing seamless and reliable services worldwide.
    """)
elif menu == "Business Problem":
    st.sidebar.markdown("""
    **Business Challenge:**  
    Predicting delivery times accurately to optimize logistics, improve customer experience, and reduce operational costs.
    """)
elif menu == "Purpose of this Website":
    st.sidebar.markdown("""
    **Purpose:**  
    To provide a user-friendly platform that estimates delivery times based on multiple operational factors using advanced machine learning.
    """)

if menu == "Predict Delivery Time":
    st.markdown(
        """
        <div style="text-align:center; padding-bottom: 20px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/cc/Uber_logo_2018.png" width="180" alt="Uber Logo">
            <h1 style="color:#1DB954; font-weight: 800;">Uber Food Delivery Time Predictor</h1>
            <p style="font-size:18px; color:#b3b3b3;">Powered by XGBoost Machine Learning Model</p>
        </div>
        """, unsafe_allow_html=True
    )

    with st.form("delivery_form"):
        st.subheader("Input Delivery Details")
        distance_km = st.number_input("Distance (km)", min_value=0.0, format="%.2f", help="Distance between restaurant and delivery address.")
        prep_time = st.number_input("Preparation Time (minutes)", min_value=0, help="Time taken to prepare the order.")
        courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, format="%.1f", help="Years of delivery courier experience.")

        weather = st.selectbox("Weather Condition", ['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'])
        traffic = st.selectbox("Traffic Level", ['Low', 'Medium', 'High'])
        time_of_day = st.selectbox("Time of Day", ['Afternoon', 'Evening', 'Night', 'Morning'])
        vehicle = st.selectbox("Vehicle Type", ['Scooter', 'Bike', 'Car'])

        submit = st.form_submit_button("Predict Delivery Time")

    if submit:
        # Prepare input dictionary
        data = {
            'Distance_km': distance_km,
            'Preparation_Time_min': prep_time,
            'Courier_Experience_yrs': courier_exp,
            'Weather_Windy': 0, 'Weather_Clear': 0, 'Weather_Foggy': 0, 'Weather_Rainy': 0, 'Weather_Snowy': 0,
            'Traffic_Level_Low': 0, 'Traffic_Level_Medium': 0, 'Traffic_Level_High': 0,
            'Time_of_Day_Afternoon': 0, 'Time_of_Day_Evening': 0, 'Time_of_Day_Night': 0, 'Time_of_Day_Morning': 0,
            'Vehicle_Type_Scooter': 0, 'Vehicle_Type_Bike': 0, 'Vehicle_Type_Car': 0
        }

        data[f'Weather_{weather}'] = 1
        data[f'Traffic_Level_{traffic}'] = 1
        data[f'Time_of_Day_{time_of_day}'] = 1
        data[f'Vehicle_Type_{vehicle}'] = 1

        input_df = pd.DataFrame([data])
        numeric_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Prediction
        prediction = model.predict(input_df)[0]
        st.markdown(f"<h2 style='color:#1DB954; text-align:center;'>Estimated Delivery Time: {prediction:.2f} minutes</h2>", unsafe_allow_html=True)
