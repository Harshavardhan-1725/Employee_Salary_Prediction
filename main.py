# main.py

import streamlit as st
import pandas as pd
import joblib

# ✅ Load model and column names using joblib (NOT cloudpickle)
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")  # Load your model
    columns = joblib.load("model_columns.pkl")  # Load expected input features
    return model, columns

# Load model and feature columns
model, model_columns = load_model()

# Streamlit App UI
st.title("💼 Employee Salary Classification App")
st.write("Predict whether an employee earns >50K or ≤50K based on input features.")

st.subheader("🔎 Input Data")

# Input form
age = st.number_input("Age", min_value=18, max_value=100)
workclass = st.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"
])
education = st.selectbox("Education", [
    "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
    "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"
])
occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
])
hours_per_week = st.slider("Hours per Week", 1, 100, 40)

# Predict
if st.button("Predict Salary"):
    # Prepare user input as DataFrame
    input_data = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "occupation": occupation,
        "hours_per_week": hours_per_week
    }

    input_df = pd.DataFrame([input_data])

    # One-hot encode and align columns
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_encoded)[0]

    st.success(f"💰 Predicted Income: **{prediction}**")
