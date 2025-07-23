import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from io import BytesIO
from streamlit_extras.avatar import avatar

# Load model and expected column structure
model = joblib.load("best_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Translations dictionary
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
            "upload_csv": "CSV अपलोड करें (थोक भविष्यानुमान)",
            "download": "⬇️ परिणाम डाउनलोड करें",
        },
        "te": {
            "title": "ఉద్యోగి జీతం వర్గీకరణ",
            "age": "వయస్సు",
            "workclass": "పని తరగతి",
            "education": "విద్య",
            "occupation": "ఉద్యోగం",
            "hours": "వారం గంటలు",
            "predict": "🎯 జీతంని ఊహించండి",
            "predicted_income": "అంచనా జీతం",
            "upload_csv": "CSV అప్లోడ్ (బల్క్ ప్రిడిక్షన్)",
            "download": "⬇️ ఫలితాలన్ని డాఉన్లోడ్ చెయండి",
        },
    }
    return translations.get(lang, translations["en"])

# Top Header with Avatar
col1, col2 = st.columns([1, 8])
with col1:
    avatar("user")
with col2:
    st.markdown("<h1 style='color:#2196F3; font-size: 40px;'>AI-Powered Salary Predictor 💼</h1>", unsafe_allow_html=True)

# Language Selector below heading
lang_choice = st.selectbox("🌐 Select Language / भाषा / భాష", ["en", "hi", "te"],
                           format_func=lambda x: {"en": "English", "hi": "Hindi", "te": "Telugu"}[x])
lang = get_translations(lang_choice)

# Predict button styling
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

# Tabs: Manual + Bulk
manual_tab, upload_tab = st.tabs(["📋 Manual Input", "📁 Bulk Upload"])

# --- Manual Input ---
with manual_tab:
    st.subheader("🔎 Manual Input")

    age = st.number_input(lang["age"], 18, 100)
    workclass = st.selectbox(lang["workclass"], ["Private 🏢", "Self-emp 🔧", "Government 🏛️"])
    education = st.selectbox(lang["education"], ["Bachelors 🎓", "HS-grad 🏫", "Masters 🎓"])
    occupation = st.selectbox(lang["occupation"], ["Tech-support 💻", "Craft-repair 🔨", "Sales 💼"])
    hours = st.slider(lang["hours"], 1, 100, 40)

    if st.button(lang["predict"]):
        # Preprocess
        input_dict = {
            "age": age,
            "workclass": workclass.split()[0],
            "education": education.split()[0],
            "occupation": occupation.split()[0],
            "hours_per_week": hours
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]
        st.success(f"{lang['predicted_income']}: **{prediction}**")

        # Download result
        csv_result = f"age,workclass,education,occupation,hours_per_week,prediction\n{age},{workclass},{education},{occupation},{hours},{prediction}"
        st.download_button("⬇️ Download This Result", csv_result, "single_prediction.csv", "text/csv")

        # Charts
        st.subheader("📊 Visual Insights")
        st.plotly_chart(px.pie(input_df, names='workclass', title="Workclass Distribution"), use_container_width=True)
        st.plotly_chart(px.bar(input_df, x='occupation', y='hours_per_week', title="Occupation vs Hours"), use_container_width=True)
        st.plotly_chart(px.histogram(input_df, x='age', nbins=10, title="Age Distribution"), use_container_width=True)

# --- Bulk Upload ---
with upload_tab:
    st.subheader(f"📁 {lang['upload_csv']}")
    uploaded_file = st.file_uploader("", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df_encoded = pd.get_dummies(df).reindex(columns=model_columns, fill_value=0)
        df["Prediction"] = model.predict(df_encoded)

        st.success("✅ Predictions Completed")
        st.dataframe(df)

        st.subheader("📊 Charts from Bulk Data")
        st.plotly_chart(px.histogram(df, x='age', nbins=10, title="Age Distribution"), use_container_width=True)

        if 'occupation' in df:
            st.plotly_chart(px.bar(df, x='occupation', color='occupation', title="Occupation Count"), use_container_width=True)

        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        st.download_button(lang["download"], buffer.getvalue(), "predicted_results.csv", "text/csv")
