import joblib
import pandas as pd

# Load model and scaler
model = joblib.load(r'H:\Kuliah\Bootcamp\Take Home Test\xgb_tuned_model.pkl')
scaler = joblib.load(r'H:\Kuliah\Bootcamp\Take Home Test\scaler.pkl')

# Get the exact feature names and order expected by the model
expected_columns = model.feature_names_in_.tolist()

# Identify numeric features (must match training data)
numeric_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

def make_prediction(distance_km, prep_time, courier_exp, weather, traffic, time_of_day, vehicle):
    # Initialize all features to 0 (scalar, not list)
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

    # Convert to DataFrame
    input_df = pd.DataFrame([data], columns=expected_columns)

    # Scale only numeric features
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # Make prediction
    prediction = model.predict(input_df)[0]
    return prediction

# --- User Input Section ---
print("Enter the following details to predict delivery time:")

try:
    distance_km = float(input("Distance (km): "))
    prep_time = int(input("Preparation Time (minutes): "))
    courier_exp = float(input("Courier Experience (years): "))

    print("Weather options: Clear, Foggy, Rainy, Snowy, Windy")
    weather = input("Weather: ").capitalize()

    print("Traffic options: Low, Medium, High")
    traffic = input("Traffic Level: ").capitalize()

    print("Time of Day options: Morning, Afternoon, Evening, Night")
    time_of_day = input("Time of Day: ").capitalize()

    print("Vehicle options: Bike, Car, Scooter")
    vehicle = input("Vehicle Type: ").capitalize()

    predicted_time = make_prediction(distance_km, prep_time, courier_exp, weather, traffic, time_of_day, vehicle)
    print(f"\nEstimated Delivery Time: {predicted_time:.2f} minutes")

except Exception as e:
    print("\nAn error occurred during prediction:")
    print(e)
