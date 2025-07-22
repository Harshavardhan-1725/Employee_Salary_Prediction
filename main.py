import streamlit as st
import pandas as pd
import joblib
import base64

# Page config
st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")

# Load model and columns
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# --- CSS for Styling ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #0f2027;
        background-image: linear-gradient(to right, #2c5364, #203a43, #0f2027);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    .stButton>button {
        background-color: #00c9a7;
        color: white;
        padding: 0.6em 2em;
        border-radius: 30px;
        border: none;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #02a188;
        transform: scale(1.05);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }

    .result-box {
        background: linear-gradient(135deg, #00c9a7, #92fe9d);
        color: black;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown("<h1 style='text-align: center;'>💼 Employee Salary Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Predict if an employee earns more than 50K or not</p>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", 18, 100, 30)
    education = st.selectbox("🎓 Education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
        "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"
    ])
    hours_per_week = st.slider("🕒 Hours per Week", 1, 100, 40)

with col2:
    workclass = st.selectbox("🏢 Workclass", [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ])
    occupation = st.selectbox("🛠️ Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
        "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
        "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces"
    ])

st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction ---
if st.button("🔮 Predict Salary"):
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
    
    try:
        proba = model.predict_proba(input_encoded)[0]
        confidence = round(max(proba) * 100, 2)
    except:
        confidence = None

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    if prediction == ">50K":
        st.markdown('<div class="result-box">💰 Predicted Income: >50K</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box">🧾 Predicted Income: ≤50K</div>', unsafe_allow_html=True)

    if confidence:
        st.write(f"📊 Confidence: **{confidence}%**")
    
    st.markdown('</div>', unsafe_allow_html=True)

