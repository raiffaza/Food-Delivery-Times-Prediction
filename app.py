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
            background-color: #00B14F; /* Uber Eats green */
            color: white;
            font-weight: 700;
            border-radius: 8px;
            height: 45px;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #22d26b;
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
            color: #00B14F;
        }
        .stAlert {
            background-color: #00B14F !important;
            color: #000000 !important;
            font-weight: 600 !important;
        }
        /* Centered content container */
        .centered-content {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
            margin-top: 50px;
            margin-bottom: 50px;
        }
        .centered-text {
            max-width: 600px;
            font-size: 20px;
            line-height: 1.6;
            text-align: center;
            color: #e0e0e0;
        }
        .blunt-image {
            border-radius: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.6);
            max-width: 400px;
            width: 100%;
            height: auto;
        }
        .logo-center {
            display: flex;
            justify-content: center;
            margin-top: 40px;
            margin-bottom: 40px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# Load model and scaler
model = joblib.load('xgb_tuned_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load images (make sure these image files are in your project folder)
logo_img = Image.open("uber eats.png")
company_profile_img = Image.open("uber eats company profile.jpeg")
business_problem_img = Image.open("uber eats business problem.jpeg")
purpose_website_img = Image.open("uber eats company profile.jpeg")  # same image as company profile

# Sidebar navigation
st.sidebar.title("Uber Eats Delivery Insights")
menu = st.sidebar.radio("Navigate", ["Company Profile", "Business Problem", "Purpose of this Website", "Predict Delivery Time"])

# Sidebar content with centered image and text side by side
if menu == "Company Profile":
    st.markdown("<div class='logo-center'>", unsafe_allow_html=True)
    st.image(logo_img, width=180)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='centered-content'>
            <img src="data:image/jpeg;base64,{company_profile_img._repr_png_().decode('utf-8')}" alt="Company Profile" class="blunt-image"/>
            <p class='centered-text'>
                Uber Eats is a global leader in food delivery, connecting customers with their favorite restaurants through innovative technology.
                Our mission is to make eating effortless and enjoyable by delivering food quickly and reliably across cities worldwide.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif menu == "Business Problem":
    st.markdown("<div class='logo-center'>", unsafe_allow_html=True)
    st.image(logo_img, width=180)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='centered-content'>
            <img src="data:image/jpeg;base64,{business_problem_img._repr_png_().decode('utf-8')}" alt="Business Problem" class="blunt-image"/>
            <p class='centered-text'>
                Accurately predicting delivery times remains a key challenge in food delivery logistics.
                Late or inaccurate estimates reduce customer satisfaction, increase operational costs, and impact brand reputation.
                Understanding and forecasting delivery time based on factors like weather, traffic, and courier experience is critical for operational efficiency.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif menu == "Purpose of this Website":
    st.markdown("<div class='logo-center'>", unsafe_allow_html=True)
    st.image(logo_img, width=180)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='centered-content'>
            <img src="data:image/jpeg;base64,{purpose_website_img._repr_png_().decode('utf-8')}" alt="Purpose of Website" class="blunt-image"/>
            <p class='centered-text'>
                This platform allows Uber Eats operations teams to input delivery parameters and instantly get a data-driven delivery time estimate.
                Powered by a machine learning model, it supports smarter decision-making, better resource allocation, and improved customer experience.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Predict Delivery Time Page
if menu == "Predict Delivery Time":
    st.markdown("<div class='logo-center'>", unsafe_allow_html=True)
    st.image(logo_img, width=180)
    st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown(f"<h2 style='color:#00B14F; text-align:center;'>Estimated Delivery Time: {prediction:.2f} minutes</h2>", unsafe_allow_html=True)
