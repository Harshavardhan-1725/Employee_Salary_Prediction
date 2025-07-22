import streamlit as st
import pandas as pd
import cloudpickle

# ✅ Load trained model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return cloudpickle.load(f)

model = load_model()

st.set_page_config(page_title="Salary Classifier", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# ✅ Input from sidebar
st.sidebar.header("Enter Employee Details")
age = st.sidebar.slider("Age", 17, 75, 30)
educational_num = st.sidebar.slider("Education Number", 1, 16, 10)
workclass = st.sidebar.slider("Workclass (Encoded)", 0, 8, 3)
occupation = st.sidebar.slider("Occupation (Encoded)", 0, 13, 4)
relationship = st.sidebar.slider("Relationship (Encoded)", 0, 5, 2)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
race = st.sidebar.slider("Race (Encoded)", 0, 4, 1)
native_country = st.sidebar.slider("Native Country (Encoded)", 0, 40, 10)
marital_status = st.sidebar.slider("Marital Status (Encoded)", 0, 6, 2)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

# ✅ Gender encoding
gender_encoded = 1 if gender == "Male" else 0

# ✅ Input DataFrame (must match trained model)
input_df = pd.DataFrame([[
    age,
    workclass,
    marital_status,
    occupation,
    relationship,
    race,
    gender_encoded,
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country,
    educational_num
]], columns=[
    'age',
    'workclass',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'educational-num'
])

st.write("🔍 Input Preview:")
st.write(input_df)

# ✅ Prediction button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)  # Model expects these 13 features
    result = ">50K" if prediction[0] == 1 else "≤50K"
    st.success(f"💡 Prediction: Employee earns {result}")
