import streamlit as st
import pandas as pd
import joblib
import json
import requests

# Custom CSS Styling
def load_custom_css():
    st.markdown("""
        <style>
            .main {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 15px;
            }
            .title {
                font-size: 36px;
                font-weight: bold;
                color: #4b4bfb;
            }
            .subtitle {
                font-size: 20px;
                color: #555;
            }
            .result {
                font-size: 28px;
                font-weight: bold;
                color: green;
            }
            .stButton>button {
                background-color: #4b4bfb;
                color: white;
                font-weight: bold;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

# Lottie Animation Loader
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Model and Columns
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()
load_custom_css()

# Title Section with Animation
col1, col2 = st.columns([1, 2])
with col1:
    lottie = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_tijmpn5v.json")
    st_lottie = st.components.v1.html(f"""
        <lottie-player src="https://assets3.lottiefiles.com/packages/lf20_tijmpn5v.json" background="transparent"
         speed="1" style="width: 250px; height: 250px;" loop autoplay></lottie-player>
    """, height=250)
with col2:
    st.markdown("<div class='main'><div class='title'>💼 Employee Salary Classification</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predict whether an employee earns >50K or ≤50K based on input details.</div></div>", unsafe_allow_html=True)

# Input Fields
st.subheader("🔎 Input Data")

age = st.number_input("👤 Age", min_value=18, max_value=100, step=1)
workclass = st.selectbox("🏢 Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"
])
education = st.selectbox("🎓 Education", [
    "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
    "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"
])
occupation = st.selectbox("🛠️ Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
])
hours_per_week = st.slider("⏱️ Hours per Week", 1, 100, 40)

# Predict Button
if st.button("🚀 Predict Salary"):
    input_data = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "occupation": occupation,
        "hours_per_week": hours_per_week
    }
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    
    if prediction == ">50K":
        st.success("💰 Predicted Income: **>50K** ✅")
    else:
        st.warning("💵 Predicted Income: **≤50K** ❗")

