import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.graph_objects as go

# Load model and columns
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# Load animation from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Add modern CSS styling
def custom_css():
    st.markdown("""
        <style>
            .main-panel {
                background: linear-gradient(135deg, #e0f7fa, #ffffff);
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .stButton>button {
                background-color: #3b5998;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #1c2d70;
                transform: scale(1.05);
            }
            .title {
                font-size: 42px;
                font-weight: 800;
                color: #2c3e50;
            }
            .subtitle {
                font-size: 20px;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

custom_css()

# Header
st.columns([1, 8, 1])[1].markdown("<div class='title'>💼 Employee Salary Predictor</div>", unsafe_allow_html=True)
st.columns([1, 8, 1])[1].markdown("<div class='subtitle'>Predict if an employee earns >50K or ≤50K with confidence chart</div>", unsafe_allow_html=True)

st.markdown("<div class='main-panel'>", unsafe_allow_html=True)

# Input form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", 18, 100, 30)
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

# Predict
st.markdown("### 📊 Prediction Result")

if st.button("🚀 Predict Salary"):
    input_data = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "occupation": occupation,
        "hours_per_week": hours_per_week
    }

    df = pd.DataFrame([input_data])
    encoded = pd.get_dummies(df)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(encoded)[0]
    probability = model.predict_proba(encoded)[0]  # returns [P(<=50K), P(>50K)]

    label = ">50K" if prediction == ">50K" else "≤50K"
    st.success(f"💰 Predicted Income: **{label}**")

    # Chart
    st.markdown("#### 🔍 Prediction Confidence")
    fig = go.Figure(data=[go.Pie(
        labels=["≤50K", ">50K"],
        values=probability,
        hole=0.4,
        marker=dict(colors=['#f94144', '#43aa8b']),
        hoverinfo="label+percent",
        textinfo="label+percent"
    )])
    fig.update_layout(
        showlegend=True,
        height=400,
        margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
