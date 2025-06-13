# Food Delivery Time Prediction

This repository contains a machine learning model to predict food delivery times based on various influencing factors such as distance, weather conditions, traffic levels, time of day, and courier experience. The goal is to optimize logistics and improve customer satisfaction by providing accurate estimated delivery times (ETAs).

## Project Overview

Food delivery services are heavily reliant on accurate delivery time estimates. This project leverages machine learning algorithms, including XGBoost and Random Forest, to predict delivery times for food orders. By training a model on a dataset containing key features such as distance, weather, traffic conditions, and more, this project aims to enhance operational efficiency and reduce delivery failures.

### Business Problem

* **How can we accurately predict food delivery time based on factors like distance, weather, traffic, time of day, vehicle type, preparation time, and courier experience?**

### Why It Matters

* **Improved Customer Satisfaction**: Accurate delivery time estimates help reduce customer frustration and increase trust in the service.
* **Enhanced Operational Efficiency**: Better predictions optimize courier allocation, manage peak-hour demand, and improve route planning.
* **Reduced Delivery Failures**: Proactively adjusting expectations based on environmental factors such as weather and traffic can minimize delays.
* **Dynamic Pricing**: Accurate predictions help with surge pricing during high-demand or adverse conditions.
* **Competitive Advantage**: Reliable ETAs provide a unique selling point in the competitive food delivery market.

## Dataset

The dataset used for this project is titled **"Food Delivery Time Prediction"** and is sourced from [Kaggle](https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction).

### Key Features:

* **Order\_ID**: Unique identifier for each order.
* **Distance\_km**: The delivery distance in kilometers.
* **Weather**: Weather conditions (Clear, Rainy, Snowy, Foggy, Windy).
* **Traffic\_Level**: Traffic conditions (Low, Medium, High).
* **Time\_of\_Day**: Time of the delivery (Morning, Afternoon, Evening, Night).
* **Vehicle\_Type**: Type of vehicle used for delivery (Bike, Scooter, Car).
* **Preparation\_Time\_min**: Time required to prepare the order (in minutes).
* **Courier\_Experience\_yrs**: Experience of the courier (in years).
* **Delivery\_Time\_min**: Total delivery time (target variable).

## Data Cleaning

* **Handling Duplicated Data**: No duplicates in the dataset.
* **Outlier Handling**: Checked for outliers, and the dataset is free from them.
* **Handling Missing Values**: Used mode, random imputation, and median strategies to handle missing values across various features.

## Exploratory Data Analysis (EDA)

The dataset reveals key insights:

* **Distance** is positively correlated with delivery time.
* **Weather conditions** significantly impact delivery time, with snowy weather causing the longest delays.
* **Time of day** affects delivery times, with afternoon deliveries generally being slower.
* **Courier experience** also plays a significant role, with more experienced couriers delivering faster.

## Machine Learning Models

### Model Candidates:

* **XGBoost** (XGBRegressor)
* **Random Forest**

Both models were evaluated, with **XGBoost** showing superior performance, yielding an MAE of **6.53 minutes** after hyperparameter tuning. This performance was more reliable than Random Forest, which had an MAE of **7.48 minutes**. That's why we gonna use **XGBoost** for our model prediction.

### Model Performance (MAE):

* **XGBoost**:

  * MAE (Before Tuning): 7.77 minutes
  * MAE (After Tuning): 6.53 minutes
* **Random Forest**:

  * MAE (Before Tuning): 6.88 minutes
  * MAE (After Tuning): 7.51 minutes

### Feature Importance (XGBoost):

* **Distance\_km**: Most important predictor.
* **Preparation\_Time\_min**: Significant but secondary.
* **Traffic\_Level\_Low**: Impacts delivery time significantly, especially in low traffic.

## Deployment

This model is deployed using **Streamlit** on Streamlit Cloud. You can interact with the model through a simple web app to predict food delivery times based on real-world conditions.

### Live Demo

* **Food Delivery Time Prediction App**:
  [Streamlit App](https://uber-delivery-times-prediction.streamlit.app/)

### How to Deploy the App on Streamlit Cloud

If you want to deploy this app on Streamlit Cloud, follow these steps:

1. **Push the Code to GitHub**:

   * Ensure the project is uploaded to a **GitHub repository**. You can push your code using the following commands:

     ```bash
     git init
     git remote add origin <your-repository-url>
     git add .
     git commit -m "Initial commit"
     git push -u origin main
     ```

2. **Go to Streamlit Cloud**:
   Visit [Streamlit Cloud](https://streamlit.io/cloud) and log in with your GitHub account.

3. **Create a New App**:

   * Click on **New app**.
   * Select the repository where your app is located.
   * Choose the branch and the entry file for your app (in my case, the file name is `app.py`).

4. **Requirements**:

   * Ensure your project has a **`requirements.txt`** file that lists the necessary dependencies. If not, create it using:

     ```bash
     pip freeze > requirements.txt
     ```
   * Common dependencies might include:

     ```text
     pandas
     numpy
     scikit-learn
     xgboost
     streamlit
     joblib
     ```

5. **Deploy**:

   * Click **Deploy**, and Streamlit Cloud will automatically build and deploy your app.
   * Once deployed, you'll receive a public URL to access the app.

### Requirements

* **Pandas**: Data manipulation.
* **NumPy**: Numerical operations.
* **Scikit-learn**: Machine learning algorithms.
* **XGBoost**: XGBoost model.
* **Streamlit**: Web app deployment.
* **Joblib**: Model serialization.


## Acknowledgements

* [Kaggle Dataset](https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction)
* [Yalçinkaya, E., & Areta Hızıroğlu, O. (2024)](https://doi.org/10.1007/s11423-016-9446-0)
* [Ansori, M., Kusumawati, R., & Amin Hariyadi, M. (2023)](https://doi.org/10.25008/ijadis.v4i2.1281)

---


