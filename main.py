import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from io import BytesIO
import base64

st.set_page_config(page_title="💼 Employee Salary Classifier", layout="wide", page_icon="💰")

# Load model and column names
@st.cache_data

def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

st.markdown("""
    <style>
    .main {
        background-color: #f9fbfd;
    }
    .css-1d391kg, .css-1v0mbdj p {
        font-family: 'Segoe UI', sans-serif;
        font-size: 18px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #135e96;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Employee Salary Classification App")
st.write("Predict whether an employee earns >50K or ≤50K based on input features.")

col1, col2 = st.columns([2, 2])

with col1:
    st.subheader("🔎 Input Employee Details")
    age = st.slider("Age", 18, 100, 30)
    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
    education = st.selectbox("Education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
        "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"])
    occupation = st.selectbox("Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
        "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
        "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces"])
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

    if st.button("🎯 Predict Salary"):
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

        if prediction == '>50K':
            st.success("💰 Predicted Income: **>50K**")
        else:
            st.warning("💸 Predicted Income: **≤50K**")

with col2:
    st.subheader("📁 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
        df['prediction'] = model.predict(df_encoded)

        st.success("✅ Predictions Done!")
        st.dataframe(df.head())

        # Downloadable CSV
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df)
        st.download_button("⬇️ Download Results as CSV", csv, "salary_predictions.csv", "text/csv")

        # Plot charts
        st.subheader("📊 Insights")
        pie_fig = px.pie(df, names='education', title='Education Level Distribution')
        st.plotly_chart(pie_fig, use_container_width=True)

        bar_fig = px.histogram(df, x='prediction', color='prediction', title='Prediction Count')
        st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Salary Classifier")
