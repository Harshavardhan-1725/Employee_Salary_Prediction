import streamlit as st
import pandas as pd
import plotly.express as px
import joblib  # <-- Changed from pickle to joblib
from io import BytesIO
from streamlit_extras.avatar import avatar
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

# Load model and columns
model = joblib.load("best_model.pkl")  # <-- Updated to use joblib
model_columns = joblib.load(open("model_columns.pkl", "rb"))

# Language support
def get_translations(lang):
    translations = {
        "en": {
            "title": "Employee Salary Classification",
            "age": "Age",
            "workclass": "Workclass",
            "education": "Education",
            "occupation": "Occupation",
            "hours": "Hours per Week",
            "predict": "🎯 Predict Salary",
            "predicted_income": "Predicted Income",
            "upload_csv": "Upload CSV for Bulk Predictions",
            "download": "⬇️ Download Results",
        },
        "hi": {
            "title": "कर्मचारी वेतन वर्गीकरण",
            "age": "आयु",
            "workclass": "कार्य वर्ग",
            "education": "शिक्षा",
            "occupation": "पेशा",
            "hours": "प्रति सप्ताह घंटे",
            "predict": "🎯 वेतन का पूर्वानुमान करें",
            "predicted_income": "अनुमानित आय",
            "upload_csv": "CSV अपलोड करें (थोक भविष्यवाणी)",
            "download": "⬇️ परिणाम डाउनलोड करें",
        },
        "te": {
            "title": "ఉద్యోగి జీతం వర్గీకరణ",
            "age": "వయస్సు",
            "workclass": "పని తరగతి",
            "education": "విద్య",
            "occupation": "ఉద్యోగం",
            "hours": "వారం గంటలు",
            "predict": "🎯 జీతాన్ని ఊహించండి",
            "predicted_income": "అంచనా జీతం",
            "upload_csv": "CSV అప్లోడ్ (బల్క్ ప్రిడిక్షన్)",
            "download": "⬇️ ఫలితాలను డౌన్‌లోడ్ చేయండి",
        },
    }
    return translations.get(lang, translations["en"])

lang_choice = st.sidebar.selectbox("🌐 Select Language / भाषा / భాష", ["en", "hi", "te"], format_func=lambda x: {"en": "English", "hi": "Hindi", "te": "Telugu"}[x])
lang = get_translations(lang_choice)

# Avatar and animated header
avatar("user")
st.markdown("<h1 style='text-align:center; color:#2196F3;'>AI-Powered Salary Predictor 💼</h1>", unsafe_allow_html=True)

# CSS for animated Predict Salary button
st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(to right, #2196F3, #21CBF3);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        transition: all 0.3s ease-in-out;
        margin-top: 10px;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Tabs Layout
manual_tab, upload_tab = st.tabs(["📋 Manual Input", "📁 Bulk Upload"])

# --- Manual Input Tab ---
with manual_tab:
    st.subheader("🔎 Manual Input")
    age = st.number_input(lang["age"], 18, 100)
    workclass = st.selectbox(lang["workclass"], ["Private 🏢", "Self-emp 🔧", "Government 🏛️"])
    education = st.selectbox(lang["education"], ["Bachelors 🎓", "HS-grad 🏫", "Masters 🎓"])
    occupation = st.selectbox(lang["occupation"], ["Tech-support 💻", "Craft-repair 🔨", "Sales 💼"])
    hours_per_week = st.slider(lang["hours"], 1, 100, 40)

    if st.button(lang["predict"]):
        input_data = {
            "age": age,
            "workclass": workclass.split(" ")[0],
            "education": education.split(" ")[0],
            "occupation": occupation.split(" ")[0],
            "hours_per_week": hours_per_week
        }
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]
        st.success(f"{lang['predicted_income']}: **{prediction}**")

        st.subheader("📊 Visual Insights")
        st.plotly_chart(px.pie(input_df, names='workclass', title="Workclass Distribution", color_discrete_sequence=['#2196F3']).update_layout(template='plotly_white'), use_container_width=True)
        st.plotly_chart(px.bar(input_df, x='occupation', y='hours_per_week', title="Occupation vs Hours", color_discrete_sequence=['#2196F3']).update_layout(template='plotly_white'), use_container_width=True)
        st.plotly_chart(px.histogram(input_df, x='age', nbins=10, title="Age Distribution", color_discrete_sequence=['#2196F3']).update_layout(template='plotly_white'), use_container_width=True)

# --- Bulk Upload Tab ---
with upload_tab:
    st.subheader(f"📁 {lang['upload_csv']}")
    uploaded_file = st.file_uploader("", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
        predictions = model.predict(df_encoded)
        df['Prediction'] = predictions

        st.success("✅ Predictions completed")
        st.dataframe(df)

        st.subheader("📊 Charts from Bulk Data")
        st.plotly_chart(px.histogram(df, x='age', nbins=10, title="Age Distribution", color_discrete_sequence=['#2196F3']).update_layout(template='plotly_white'), use_container_width=True)
        if 'occupation' in df:
            st.plotly_chart(px.bar(df, x='occupation', color='occupation', title="Occupation Count", color_discrete_sequence=['#2196F3']).update_layout(template='plotly_white'), use_container_width=True)

        output = BytesIO()
        df.to_csv(output, index=False)
        st.download_button(label=lang["download"], data=output.getvalue(), file_name="predicted_results.csv", mime="text/csv")
