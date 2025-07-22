import streamlit as st
import pandas as pd
import joblib

# 🎯 Load model and feature columns
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# 🌟 Page Config
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

# 🎨 Page Title
st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>💼 Employee Salary Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether an employee earns <strong>>50K</strong> or <strong>≤50K</strong> based on key job details.</p>", unsafe_allow_html=True)
st.markdown("---")

# 🔎 Input Section
st.markdown("### 🔎 Enter Employee Information")

# Split form into two columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=18, max_value=100, value=30)
    education = st.selectbox("🎓 Education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
        "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"
    ])
    hours_per_week = st.slider("⏱️ Hours per Week", 1, 100, 40)

with col2:
    workclass = st.selectbox("🏢 Workclass", [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ])
    occupation = st.selectbox("🛠️ Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
        "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
        "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces"
    ])

st.markdown("---")

# 🚀 Predict Button
if st.button("🔍 Predict Salary"):
    input_data = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "occupation": occupation,
        "hours_per_week": hours_per_week
    }

    input_df = pd.DataFrame([input_data])

    # One-hot encoding + column alignment
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]

    # 🎉 Result
    if prediction == ">50K":
        st.success("🎉 The employee is likely to earn **>50K** 💰")
    else:
        st.info("📉 The employee is likely to earn **≤50K**")

    st.markdown("---")
    st.markdown("🔁 You can modify the inputs and re-run the prediction anytime.")

