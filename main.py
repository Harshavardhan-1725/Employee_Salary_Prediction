import streamlit as st
import pandas as pd
import joblib

# 💾 Load model and columns
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# 🌐 Page setup
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# 🎨 Gradient title using markdown
st.markdown(
    """
    <div style='text-align: center; padding: 10px; background: linear-gradient(to right, #4facfe, #00f2fe); border-radius: 10px;'>
        <h1 style='color: white;'>💼 Employee Salary Classifier</h1>
        <p style='color: white;'>Predict if income is >50K or ≤50K based on job and education details</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## 📝 Enter Employee Details")

# 🧾 Inputs in columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
    education = st.selectbox("🎓 Education Level", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
        "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"
    ])
    hours_per_week = st.slider("🕒 Weekly Working Hours", 1, 100, 40)

with col2:
    workclass = st.selectbox("🏢 Workclass Type", [
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

# 🧠 Prediction logic
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

    # 🎯 Display result with styling
    if prediction == ">50K":
        st.success("✅ **Prediction:** The employee is likely to earn more than **50K** 💰")
    else:
        st.warning("⚠️ **Prediction:** The employee is likely to earn **50K or less**")

    st.markdown("---")
    st.info("Modify the values above and click Predict again to test different employee profiles.")
