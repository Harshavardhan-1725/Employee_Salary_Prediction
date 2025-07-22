import streamlit as st
import pandas as pd
import cloudpickle

# ✅ Load model using cloudpickle
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return cloudpickle.load(f)

model = load_model()

# ✅ Streamlit App UI
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
educational_num = st.sidebar.slider("Educational Number (5 to 16)", 5, 16, 10)
workclass = st.sidebar.slider("Workclass (Encoded)", 0, 6, 2)
occupation = st.sidebar.slider("Occupation (Encoded)", 0, 13, 4)
relationship = st.sidebar.slider("Relationship (Encoded)", 0, 5, 2)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
race = st.sidebar.slider("Race (Encoded)", 0, 4, 1)
native_country = st.sidebar.slider("Native Country (Encoded)", 0, 40, 10)
marital_status = st.sidebar.slider("Marital Status (Encoded)", 0, 3, 1)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 99999, 0)

# ✅ Encode gender
gender_encoded = 1 if gender == "Male" else 0

# ✅ Create input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender_encoded],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    'educational-num': [educational_num]
})

# ✅ Ensure column order exactly matches training data
expected_cols = [
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
]
input_df = input_df[expected_cols]

# ✅ Show input
st.write("### 🔎 Input Data")
st.write(input_df)

# ✅ Make prediction
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    result = ">50K" if prediction[0] == 1 else "≤50K"
    st.success(f"✅ Prediction: Employee earns {result}")

# ✅ Batch prediction section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    # ✅ Reorder columns to match model
    batch_data = batch_data[expected_cols]

    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = [">50K" if pred == 1 else "≤50K" for pred in batch_preds]

    st.write("✅ Predictions:")
    st.write(batch_data.head())

    # ✅ Downloadable CSV
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
