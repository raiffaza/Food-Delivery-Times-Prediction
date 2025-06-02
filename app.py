import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Inject custom CSS for styling
def local_css():
    st.markdown(
        """
        <style>
        /* General styles */
        body {
            background-color: #000000;
            color: #f0f0f0;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #121212;
            color: #ffffff;
            border-right: 1px solid #333333;
        }
        h1, h2, h3, h4, h5 {
            color: #ffffff;
            font-weight: 700;
            text-align: center;
        }
        .main {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }
        .content-block {
            background-color: #121212;
            border-radius: 12px;
            padding: 3rem;
            margin: 2rem 0;
            box-shadow: 0 8px 32px rgba(0, 177, 79, 0.1);
            border: 1px solid #333333;
        }
        .content-text {
            text-align: center;
            font-size: 1.1rem;
            line-height: 1.8;
            color: #e0e0e0;
            margin: 2rem 0;
        }
        .content-image {
            border-radius: 0 !important;
            width: 100%;
            margin: 1.5rem 0;
            filter: brightness(0.95);
        }
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #00B14F;
            color: white;
            font-weight: 700;
            border-radius: 8px;
            height: 45px;
            width: 100%;
            transition: all 0.3s ease;
            border: none;
        }
        .stButton>button:hover {
            background-color: #22d26b;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 177, 79, 0.3);
        }
        .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div {
            background-color: #222222;
            color: #f0f0f0;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 12px;
        }
        .css-1kyxreq.edgvbvh3 {
            color: #00B14F;
        }
        .stAlert {
            background-color: #00B14F !important;
            color: #000000 !important;
            font-weight: 600 !important;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# Load model and scaler
model = joblib.load('xgb_tuned_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load images
logo = Image.open("uber eats.png")
company_profile_img = Image.open("uber eats company profile.jpeg")
business_problem_img = Image.open("uber eats business problem.jpeg")
purpose_img = Image.open("uber eats company profile.jpeg")

# Sidebar navigation
st.sidebar.title("Uber Eats Delivery Insights")
menu = st.sidebar.radio("Navigate", ["Company Profile", "Business Problem", "Purpose of this Website", "Predict Delivery Time"])

# Main content
if menu != "Predict Delivery Time":
    # Logo centered at top
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(logo, width=180)
    st.markdown('</div>', unsafe_allow_html=True)

# Company Profile
if menu == "Company Profile":
    st.markdown('<div class="content-block">', unsafe_allow_html=True)
    st.image(company_profile_img, use_column_width=True, output_format="auto", clamp=True, caption="", channels="RGB")
    st.markdown("""
    <div class="content-text">
    <h2>Premium Food Delivery Service</h2>
    Uber Eats redefines culinary convenience with our premium delivery platform that connects discerning customers 
    with the finest restaurants in their city. As part of the global Uber Technologies Inc., we bring unparalleled 
    expertise in logistics and technology to the food delivery space.
    <br><br>
    Our network spans over 6,000 cities across 45 countries, delivering exceptional dining experiences directly 
    to your doorstep. We partner exclusively with top-rated establishments to ensure every meal meets our exacting 
    standards of quality and presentation.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Business Problem
elif menu == "Business Problem":
    st.markdown('<div class="content-block">', unsafe_allow_html=True)
    st.image(business_problem_img, use_column_width=True, output_format="auto", clamp=True, caption="", channels="RGB")
    st.markdown("""
    <div class="content-text">
    <h2>Optimizing the Delivery Experience</h2>
    In the premium food delivery market, timing is everything. Our clients expect nothing less than perfection - 
    meals arriving at the ideal temperature and presentation, precisely when promised.
    <br><br>
    The challenge lies in accurately predicting delivery times amidst dynamic variables: urban traffic patterns, 
    weather conditions, restaurant preparation variances, and courier routing efficiency. Traditional estimation 
    methods fail to account for these complex, interacting factors, leading to suboptimal customer experiences.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Purpose of this Website
elif menu == "Purpose of this Website":
    st.markdown('<div class="content-block">', unsafe_allow_html=True)
    st.image(purpose_img, use_column_width=True, output_format="auto", clamp=True, caption="", channels="RGB")
    st.markdown("""
    <div class="content-text">
    <h2>Precision Delivery Forecasting</h2>
    This advanced analytics platform represents the cutting edge in delivery logistics technology. 
    By leveraging machine learning and real-time data integration, we provide our operations teams 
    with unparalleled predictive accuracy.
    <br><br>
    The system synthesizes multiple data streams - from historical performance metrics to current 
    environmental conditions - to generate precise delivery estimates. This enables proactive resource 
    allocation and sets a new standard for reliability in premium food delivery services.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Page
elif menu == "Predict Delivery Time":
    # Display logo and title
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(logo, width=180)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(
        "<h1 style='color:#00B14F; font-weight: 800; text-align:center;'>Uber Eats Delivery Time Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#b3b3b3; font-size:18px; text-align:center; margin-bottom: 2rem;'>Powered by XGBoost Machine Learning Model</p>",
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
        st.markdown(
            f"""
            <div style="background-color: #121212; border-radius: 12px; padding: 2rem; margin-top: 2rem; border: 1px solid #333333;">
                <h2 style='color:#00B14F; text-align:center;'>Estimated Delivery Time</h2>
                <h1 style='color:#ffffff; text-align:center; font-weight: 700;'>{prediction:.0f} minutes</h1>
                <p style='color:#b3b3b3; text-align:center;'>Premium delivery service guaranteed</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
