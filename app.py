import streamlit as st
import numpy as np
import joblib

# Load trained models
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Loan Prediction")

# Gender field: Male or Female
gender = st.selectbox('Gender', ['Male', 'Female'])
gender = 1 if gender == 'Male' else 0

# Married field: Y or N
married = st.selectbox('Married', ['Yes', 'No'])
married = 1 if married == 'Yes' else 0

# Education field: Graduate or Undergraduate
education = st.selectbox('Education', ['Graduate', 'Undergraduate'])
education = 1 if education == 'Graduate' else 0

# Self Employed field: Y or N
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
self_employed = 1 if self_employed == 'Yes' else 0

# Property Area field: Urban, Semi Urban, or Rural
property_area = st.selectbox('Property Area', ['Urban', 'Semi Urban', 'Rural'])
property_area = {'Urban': 0, 'Semi Urban': 1, 'Rural': 2}[property_area]

# Collect remaining input fields
applicant_income = st.number_input("Applicant's Income", 0, 100000, step=1000)
coapplicant_income = st.number_input("Coapplicant's Income", 0, 100000, step=1000)
loan_amount = st.number_input('Loan Amount (in thousands)', 0, 500)
loan_amount_term = st.number_input('Loan Amount Term (in months)', 12, 360, step=12)
credit_history = st.selectbox('Credit History', ['Meets Guidelines', 'Does Not Meet Guidelines'])
credit_history = 1 if credit_history == 'Meets Guidelines' else 0

# Feature Engineering
total_income = applicant_income + coapplicant_income
loan_income_ratio = loan_amount / total_income if total_income > 0 else 0

# Prepare the features for prediction
features = np.array([[gender, married, education, self_employed, property_area,
                      applicant_income, coapplicant_income, loan_amount, 
                      loan_amount_term, credit_history, total_income, loan_income_ratio]])
features = scaler.transform(features)

# Prediction button for each model
if st.button('Predict with Random Forest'):
    prediction = rf_model.predict(features)
    st.write("Loan Status Prediction (Random Forest):", "Approved" if prediction == 1 else "Not Approved")

if st.button('Predict with XGBoost'):
    prediction = xgb_model.predict(features)
    st.write("Loan Status Prediction (XGBoost):", "Approved" if prediction == 1 else "Not Approved")
