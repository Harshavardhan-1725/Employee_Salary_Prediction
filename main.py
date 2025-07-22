import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_extras.let_it_rain import rain
from streamlit_extras.avatar import avatar
from streamlit_extras.colored_header import colored_header
from googletrans import Translator

st.set_page_config(page_title="💼 Salary Classifier", page_icon="💼", layout="centered")
translator = Translator()

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
avatar(name="👩‍💼", size=100)
st.markdown("### 👋 Welcome to the Employee Salary Predictor")
st.markdown('<div class="animated-title">🔮 Smart AI Tool to Predict Salaries</div>', unsafe_allow_html=True)

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
        "welcome": "Welcome"
    },
    "Hindi": {
        "title": "💼 कर्मचारी वेतन वर्गीकरण",
        "predict_button": "✨ वेतन का अनुमान लगाएं",
        "predicted_income": "💰 अनुमानित आय: ",
        "upload_csv": "📁 CSV अपलोड करें (थोक पूर्वानुमान के लिए)",
        "download_button": "⬇️ CSV डाउनलोड करें",
        "welcome": "स्वागत है"
    },
    "Telugu": {
        "title": "💼 ఉద్యోగి జీతం వర్గీకరణ",
        "predict_button": "✨ జీతం అంచనా వేయండి",
        "predicted_income": "💰 అంచనా జీతం: ",
        "upload_csv": "📁 CSV అప్లోడ్ చేయండి (బల్క్ పూర్వానుమానానికి)",
        "download_button": "⬇️ CSV డౌన్లోడ్ చేయండి",
        "welcome": "స్వాగతం"
    }
}

st.title(labels[lang]["title"])

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

    rain(emoji="💸", font_size=30, falling_speed=5, animation_length="infinite")
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
    st.write("🔍 Preview of uploaded data:", df.head())

    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    predictions = model.predict(df_encoded)
    df["Prediction"] = predictions
    st.session_state.bulk_df = df

    st.success("✅ Predictions completed!")
    st.dataframe(df)

    pie_chart = px.pie(df, names="education", title="Education Level Distribution")
    pred_bar = px.bar(df["Prediction"].value_counts().reset_index(), x="index", y="Prediction",
                      title="Prediction Counts", labels={"index": "Income"})

    st.plotly_chart(pie_chart)
    st.plotly_chart(pred_bar)

    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button(labels[lang]["download_button"], csv_download, "predictions.csv", "text/csv")
