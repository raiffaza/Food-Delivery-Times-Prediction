import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Inject custom CSS for styling and layout
def local_css():
    st.markdown(
        """
        <style>
        /* General styles */
        body {
            background-color: #0f0f0f;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #121212;
            color: #ffffff;
        }
        h1, h2, h3, h4, h5 {
            color: #ffffff;
            font-weight: 700;
        }
        .logo-center {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .logo-center img {
            width: 120px;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 6px 16px rgba(0, 177, 79, 0.3);
        }
        .image-blunt {
            width: 100%;
            max-width: 450px;
            border-radius: 20px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6);
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        .text-center {
            max-width: 700px;
            margin: 30px auto;
            font-size: 1.2rem;
            line-height: 1.6;
            text-align: center;
            color: #cccccc;
        }
        .stButton>button {
            background-color: #00B14F;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            height: 45px;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #22d26b;
            color: #000;
        }
        .stTextInput>div>input, .stNumberInput>div>input {
            background-color: #1e1e1e;
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 10px;
        }
        .css-1kyxreq.edgvbvh3 {  /* selectbox arrow */
            color: #00B14F;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# Load model and scaler
model = joblib.load('xgb_tuned_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load images (ensure these images are in your project folder)
logo_img = Image.open("uber eats.png")
company_profile_img = Image.open("uber eats company profile.jpeg")
business_problem_img = Image.open("uber eats business problem.jpeg")
purpose_website_img = Image.open("uber eats company profile.jpeg")  # same image as company profile

# Sidebar navigation
st.sidebar.title("Uber Eats Delivery Insights")
menu = st.sidebar.radio("Navigate", ["Company Profile", "Business Problem", "Purpose of this Website", "Predict Delivery Time"])

# Show small centered logo on all pages
st.markdown("<div class='logo-center'>", unsafe_allow_html=True)
st.image(logo_img)
st.markdown("</div>", unsafe_allow_html=True)

if menu == "Company Profile":
    st.image(company_profile_img, width=450)
    st.markdown("""
        <div class="text-center">
        Uber Eats is a global leader in food delivery, connecting customers with their favorite restaurants through innovative technology.
        Our mission is to make eating effortless and enjoyable by delivering food quickly and reliably across cities worldwide.
        </div>
        """, unsafe_allow_html=True)

elif menu == "Business Problem":
    st.image(business_problem_img, width=450)
    st.markdown("""
        <div class="text-center">
        Accurately predicting delivery times remains a key challenge in food delivery logistics.
        Late or inaccurate estimates reduce customer satisfaction, increase operational costs, and impact brand reputation.
        Understanding and forecasting delivery time based on factors like weather, traffic, and courier experience is critical for operational efficiency.
        </div>
        """, unsafe_allow_html=True)

elif menu == "Purpose of this Website":
    st.image(purpose_website_img, width=450)
    st.markdown("""
        <div class="text-center">
        This platform allows Uber Eats operations teams to input delivery parameters and instantly get a data-driven delivery time estimate.
        Powered by a machine learning model, it supports smarter decision-making, better resource allocation, and improved customer experience.
        </div>
        """, unsafe_allow_html=True)

elif menu == "Predict Delivery Time":
    st.markdown(
        "<h1 style='color:#00B14F; font-weight: 800; text-align:center;'>Uber Eats Delivery Time Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#b3b3b3; font-size:18px; text-align:center;'>Powered by XGBoost Machine Learning Model</p>",
        unsafe_allow_html=True
    )

    with st.form("delivery_form"):
        st.subheader("Input Delivery Details")
        distance_km = st.number_input("Distance (km)", min_value=0.0, format="%.2f", help="Distance between restaurant and delivery address.")
        prep_time = st.number_input("Preparation Time (minutes)", min_value=0, help="Time taken to prepare the order.")
        courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, format="%.1f", help="Years of delivery courier experience.")

        st.markdown("<span style='color:#00B14F; font-weight:bold;'>Weather Condition</span>", unsafe_allow_html=True)
        weather = st.selectbox("", ['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'])

        st.markdown("<span style='color:#00B14F; font-weight:bold;'>Traffic Level</span>", unsafe_allow_html=True)
        traffic = st.selectbox("", ['Low', 'Medium', 'High'])

        st.markdown("<span style='color:#00B14F; font-weight:bold;'>Time of Day</span>", unsafe_allow_html=True)
        time_of_day = st.selectbox("", ['Afternoon', 'Evening', 'Night', 'Morning'])

        st.markdown("<span style='color:#00B14F; font-weight:bold;'>Vehicle Type</span>", unsafe_allow_html=True)
        vehicle = st.selectbox("", ['Scooter', 'Bike', 'Car'])

        submit = st.form_submit_button("Predict Delivery Time")

    if submit:
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

        # Fix for scaler transform - ensure correct columns and order
        numeric_data = input_df[numeric_features]
        numeric_scaled = scaler.transform(numeric_data)
        input_df.loc[:, numeric_features] = numeric_scaled

        prediction = model.predict(input_df)[0]
        st.markdown(f"<h2 style='color:#00B14F; text-align:center;'>Estimated Delivery Time: {prediction:.2f} minutes</h2>", unsafe_allow_html=True)
