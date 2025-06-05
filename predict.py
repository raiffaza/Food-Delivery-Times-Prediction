import joblib
import pandas as pd
# Load model
model = joblib.load('xgb_tuned_model.pkl')

# Get the exact feature names and order expected by the model
expected_columns = model.feature_names_in_.tolist()

# Identify numeric features (must match training data)
numeric_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

def make_prediction(distance_km, prep_time, courier_exp, weather, traffic, time_of_day, vehicle):
    # Inisialisasi semua fitur dengan 0
    data = {col: 0 for col in expected_columns}

    # Isi fitur numerik
    data['Distance_km'] = distance_km
    data['Preparation_Time_min'] = prep_time
    data['Courier_Experience_yrs'] = courier_exp

    # Set fitur kategorikal (pastikan format nama sesuai encoding)
    for prefix, value in zip(
        ['Weather_', 'Traffic_Level_', 'Time_of_Day_', 'Vehicle_Type_'],
        [weather, traffic, time_of_day, vehicle]
    ):
        col_name = prefix + str(value)
        if col_name in data:
            data[col_name] = 1
        else:
            print(f"⚠️ Warning: Column '{col_name}' not found in model features.")

    # Konversi ke DataFrame dengan urutan kolom yang benar
    input_df = pd.DataFrame([data], columns=expected_columns)

    # Lakukan prediksi
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
