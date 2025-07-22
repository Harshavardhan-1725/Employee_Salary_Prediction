import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_extras.colored_header import colored_header

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
        "completed": "✅ Predictions completed!"
    },
    "Hindi": {
        "title": "💼 कर्मचारी वेतन वर्गीकरण",
        "predict_button": "✨ वेतन का अंकान लगाएं",
        "predicted_income": "💰 अंकानित आय: ",
        "upload_csv": "📁 CSV अपलोड करें (थोक पूर्वानुमान के लिए)",
        "download_button": "⬇️ CSV डाउनलोड करें",
        "welcome": "स्वागत है",
        "enter_details": "🔍 कर्मचारी विवरण दर्ज करें",
        "preview": "🔍 अपलोड किए गए डेटा का पूर्वावलोकन:",
        "completed": "✅ पूर्वानुमान पूरे हुए!"
    },
    "Telugu": {
        "title": "💼 ఉద్యోగి జీతం వర్గీకరణ",
        "predict_button": "✨ జీతం అంచనా వేయండి",
        "predicted_income": "💰 అంచన జీతం: ",
        "upload_csv": "📁 CSV అప్లోడ్ చేయం (బల్క్ పూర్వానుమానకు కోసంది)",
        "download_button": "⬇️ CSV డాఉన్లోడ్ చేయం",
        "welcome": "స్వాగతం",
        "enter_details": "🔍 ఉద్యోగి వివరాలు నమోదు చేయండి",
        "preview": "🔍 అప్లోడ్ చేసిన డేటా ప్రివ్యూకు:",
        "completed": "✅ అంచనాలు పూర్తయ్యాయి!"
    }
}

st.markdown(f"### 👋 {labels[lang]['welcome']} to the Employee Salary Predictor")
st.markdown(f'<div class="animated-title">🔮 Smart AI Tool to Predict Salaries</div>', unsafe_allow_html=True)
st.title(labels[lang]["title"])

# --- Single Prediction Input ---
st.subheader(labels[lang]["enter_details"])

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

if st.button(labels[lang]["predict_button"]):
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

    st.success(labels[lang]["predicted_income"] + f"**{prediction}**")

    # Show animated chart for single prediction (optional)
    st.subheader("📊 Prediction Result Chart")
    pred_df = pd.DataFrame({"Prediction": st.session_state.predictions})
    pred_chart = px.bar(pred_df.value_counts().reset_index(), x="Prediction", y="count",
                        title="Live Prediction Distribution", text="count", color="Prediction")
    st.plotly_chart(pred_chart, use_container_width=True)

# --- Show Prediction Bar Chart ---
if st.session_state.predictions:
    st.subheader("📊 Live Prediction Summary")
    pred_counts = pd.Series(st.session_state.predictions).value_counts().reset_index()
    pred_counts.columns = ["Income", "Count"]
    bar_chart = px.bar(pred_counts, x="Income", y="Count", color="Income",
                       title="Prediction Count by Class", text="Count")
    st.plotly_chart(bar_chart, use_container_width=True)

# --- Bulk Prediction from CSV ---
st.subheader(labels[lang]["upload_csv"])
uploaded_file = st.file_uploader("Upload CSV with employee records", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(labels[lang]["preview"], df.head())

    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    predictions = model.predict(df_encoded)
    df["Prediction"] = predictions
    st.session_state.bulk_df = df

    st.success(labels[lang]["completed"])
    st.dataframe(df)

    pie_chart = px.pie(df, names="education", title="Education Level Distribution")
    pred_bar = px.bar(df["Prediction"].value_counts().reset_index(), x="index", y="Prediction",
                      title="Prediction Counts", labels={"index": "Income"})

    st.plotly_chart(pie_chart)
    st.plotly_chart(pred_bar)

    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button(labels[lang]["download_button"], csv_download, "predictions.csv", "text/csv")
