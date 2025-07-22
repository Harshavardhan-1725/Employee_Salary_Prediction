import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# Set page config
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/briefcase.png", width=80)
    st.title("💼 Salary Predictor")
    st.markdown("Predict whether an employee earns **>50K** or **≤50K** based on profile.")
    st.markdown("---")
    st.caption("Developed by [Your Name]")

# Title & Instructions
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Employee Salary Classification</h1>", unsafe_allow_html=True)
st.write("Fill in the details below to predict the salary class:")

# Input form layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=18, max_value=100)
    education = st.selectbox("🎓 Education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
        "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"
    ])
    hours_per_week = st.slider("⏰ Hours/Week", 1, 100, 40)

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

# Prediction
if st.button("🔍 Predict Salary"):
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

    # Optional probability
    try:
        proba = model.predict_proba(input_encoded)[0]
        confidence = round(max(proba) * 100, 2)
    except:
        proba = None
        confidence = None

    # Display result
    if prediction == ">50K":
        st.success(f"💰 **Predicted Income: >50K USD**")
    else:
        st.info(f"🧾 **Predicted Income: ≤50K USD**")

    if confidence:
        st.write(f"📊 Confidence: `{confidence}%`")

    # Optional chart
    if proba is not None:
        st.subheader("🔢 Probability Breakdown")
        st.bar_chart({
            "≤50K": [proba[0]],
            ">50K": [proba[1]]
        })

