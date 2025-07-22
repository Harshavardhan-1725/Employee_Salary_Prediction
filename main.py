import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_extras.colored_header import colored_header
import os

st.set_page_config(page_title="💼 Salary Classifier", page_icon="💼", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
        padding: 2rem;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .animated-title {
        font-size: 30px;
        color: #333;
        animation: glow 2s ease-in-out infinite alternate;
        text-align: center;
    }
    @keyframes glow {
        from {
            text-shadow: 0 0 10px #4CAF50;
        }
        to {
            text-shadow: 0 0 20px #45a049, 0 0 30px #45a049;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model and column names
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# Initialize prediction tracking
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = pd.DataFrame()

# Profile welcome
st.markdown("""
<div style='text-align: center;'>
    <img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' width='100' style='border-radius: 50%;'/>
</div>
""", unsafe_allow_html=True)

# Language selection
lang = st.selectbox("🌐 Choose Language", ["English", "Hindi", "Telugu"])

# Translation helper
labels = {
    "English": {
        "title": "💼 Employee Salary Classification",
        "predict_button": "✨ Predict Salary",
        "predicted_income": "💰 Predicted Income: ",
        "upload_csv": "📁 Upload CSV for Bulk Prediction (Optional)",
        "download_button": "⬇️ Download Predictions as CSV",
        "welcome": "Welcome",
        "enter_details": "🔍 Enter Employee Details",
        "preview": "🔍 Preview of uploaded data:",
        "completed": "✅ Predictions completed!",
        "age": "Age",
        "workclass": "Workclass",
        "education": "Education",
        "occupation": "Occupation",
        "hours": "Hours per Week"
    },
    "Hindi": {
        "title": "💼 कर्मचारी वेतन वर्गीकरण",
        "predict_button": "✨ वेतन का अंकान लगाएं",
        "predicted_income": "💰 अंकांकित आय: ",
        "upload_csv": "📁 CSV अपलोड करें (थोक पूर्वानुमान के लिए)",
        "download_button": "⬇️ CSV डाउनलोड करें",
        "welcome": "स्वागत है",
        "enter_details": "🔍 कर्मचारी विवरण दर्ज करें",
        "preview": "🔍 अपलोड किए गए डेटा का पूर्वावलोकन:",
        "completed": "✅ पूर्वानुमान पूरे हुए!",
        "age": "उम्र",
        "workclass": "कार्यक्षेत्र",
        "education": "शिक्षा",
        "occupation": "पेशा",
        "hours": "प्रति सप्ताह घंटे"
    },
    "Telugu": {
        "title": "💼 జీతకార్మికుల జీతం వర్గీకరణ",
        "predict_button": "✨ జీతం అంచన వేయంది",
        "predicted_income": "💰 అంచినీత జీతం: ",
        "upload_csv": "📁 CSV అప్లోడ్ చేయంది (బల్క్ పూర్వానుమానకోసం)",
        "download_button": "⬇️ CSV డౌన్లోడ్ చేయంది",
        "welcome": "స్వాగతం",
        "enter_details": "🔍 ఉద్యోగి వివరాలు నమోది చేయండి",
        "preview": "🔍 అప్లోడ్ చేసిన డేటా ప్రివ్యుకు:",
        "completed": "✅ అంచనాలు పూర్త్యయాయి!",
        "age": "వయస్సు",
        "workclass": "పని తరగతి",
        "education": "విద్య",
        "occupation": "ఉద్యోగం",
        "hours": "వారం గంటలు"
    }
}

st.markdown(f"### 👋 {labels[lang]['welcome']} to the Employee Salary Predictor")
st.markdown(f'<div class="animated-title">🔮 Smart AI Tool to Predict Salaries</div>', unsafe_allow_html=True)
st.title(labels[lang]["title"])

# --- Single Prediction Input ---
st.subheader(labels[lang]["enter_details"])

age = st.number_input(labels[lang]["age"], min_value=18, max_value=100)
workclass = st.selectbox(labels[lang]["workclass"], [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education = st.selectbox(labels[lang]["education"], [
    "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
    "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"])
occupation = st.selectbox(labels[lang]["occupation"], [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"])
hours_per_week = st.slider(labels[lang]["hours"], 1, 100, 40)
