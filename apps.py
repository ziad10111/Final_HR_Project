import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained Linear Regression model and the scaler
model_lr = joblib.load('D:\\Eplsion Project\\streamlit\\HR_model.h5')
scaler = joblib.load('D:\\Eplsion Project\\streamlit\\HR_scaler.h5')

# Load expected model features
model_features = joblib.load('D:\\Eplsion Project\\streamlit\\HR_features.h5')

# Define the Streamlit app
def main():
    st.title("Employee Monthly Income Prediction")
    st.info('Predict your monthly income based on your professional details.')


    # Extracting job roles from model features
    job_roles = [feature.split('_')[1] for feature in model_features if feature.startswith('JobRole_')]
    job_roles = sorted(set(job_roles))  # Remove duplicates and sort

    # User inputs organized similarly to the bike rental app
    age = st.number_input("Enter Age:", min_value=18, max_value=65, value=30, step=1)
    education = st.number_input("Enter Education Level (1-5):", min_value=1, max_value=5, value=3, step=1)
    total_working_years = st.number_input("Enter Total Working Years:", min_value=0, max_value=40, value=10, step=1)
    job_role = st.selectbox("Select Job Role:", job_roles)
    gender = st.selectbox("Select Gender:", ['Male', 'Female'])

    if st.button('Predict Monthly Income'):
        # Create DataFrame from inputs
        input_data = pd.DataFrame([[age, education, total_working_years, job_role, gender]],
                                  columns=['Age', 'Education', 'TotalWorkingYears', 'JobRole', 'Gender'])
        input_data = pd.get_dummies(input_data)

        # Ensure all required dummy columns are present
        for column in model_features:
            if column not in input_data.columns:
                input_data[column] = 0

        input_data = input_data.reindex(columns=model_features, fill_value=0)

        # Scale the features
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model_lr.predict(input_data_scaled)
        st.metric("Predicted Monthly Income", f"${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
