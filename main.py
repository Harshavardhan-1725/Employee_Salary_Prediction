import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="💼 Salary Classifier", page_icon="💼", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
        padding: 2rem;
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

# App Header
st.title("💼 Employee Salary Classification")
st.markdown("""
Predict whether an employee earns **>50K** or **≤50K** based on input features.
""")

# --- Single Prediction Input ---
st.subheader("🔍 Enter Employee Details")

age = st.number_input("Age", min_value=18, max_value=100)
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
hours_per_week = st.slider("Hours per Week", 1, 100, 40)

if st.button("✨ Predict Salary"):
    input_data = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "occupation": occupation,
        "hours_per_week": hours_per_week
    }

    input_df = pd.DataFrame([input_data])
    encoded = pd.get_dummies(input_df)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(encoded)[0]
    st.session_state.predictions.append(prediction)

    st.success(f"💰 Predicted Income: **{prediction}**")

# --- Show Prediction Bar Chart ---
if st.session_state.predictions:
    st.subheader("📊 Live Prediction Summary")
    pred_counts = pd.Series(st.session_state.predictions).value_counts().reset_index()
    pred_counts.columns = ["Income", "Count"]
    bar_chart = px.bar(pred_counts, x="Income", y="Count", color="Income",
                       title="Prediction Count by Class", text="Count")
    st.plotly_chart(bar_chart, use_container_width=True)

# --- Bulk Prediction from CSV ---
st.subheader("📁 Upload CSV for Bulk Prediction (Optional)")
uploaded_file = st.file_uploader("Upload CSV with employee records", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("🔍 Preview of uploaded data:", df.head())

    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    predictions = model.predict(df_encoded)
    df["Prediction"] = predictions

    st.success("✅ Predictions completed!")
    st.dataframe(df)

    # Show chart for uploaded predictions
    pie_chart = px.pie(df, names="education", title="Education Level Distribution")
    pred_bar = px.bar(df["Prediction"].value_counts().reset_index(), x="index", y="Prediction", 
                      title="Prediction Counts", labels={"index": "Income"})

    st.plotly_chart(pie_chart)
    st.plotly_chart(pred_bar)

    # Downloadable result
    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions as CSV", csv_download, "predictions.csv", "text/csv")
