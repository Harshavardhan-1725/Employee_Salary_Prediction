import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from io import BytesIO

# Load model and columns
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# Multilingual options
LANGUAGES = {
    "English": {
        "title": "💼 Employee Salary Classification App",
        "intro": "Predict whether an employee earns >50K or ≤50K based on input features.",
        "age": "Age",
        "workclass": "Workclass",
        "education": "Education",
        "occupation": "Occupation",
        "hours": "Hours per Week",
        "predict": "Predict Salary",
        "upload_csv": "Upload CSV for Bulk Predictions",
        "download": "Download Results",
        "predicted_income": "💰 Predicted Income"
    },
    "Hindi": {
        "title": "💼 कर्मचारी वेतन वर्गीकरण एप्प",
        "intro": "इनपुट सुविधाओं के आधार पर अनुमान लगाएं कि कर्मचारी की आय >50K या ≤50K है।",
        "age": "आयु",
        "workclass": "कार्य वर्ग",
        "education": "शिक्षा",
        "occupation": "पेशा",
        "hours": "प्रति सप्ताह घंटे",
        "predict": "वेतन की भविष्य करें",
        "upload_csv": "CSV अपलोड करें",
        "download": "परिणाम डाउनलोड करें",
        "predicted_income": "💰 अनुमानित आय"
    },
    "Telugu": {
        "title": "💼 ఉద్యోగి జీతం వర్గీకరణ్ యాప్ప్",
        "intro": "ఇన్పుట్ ఫీచర్ల ఆధారంగా ఉద్యోగి ఆదాయం >50K లేదా ఢ6ి కా తక్కువాలని అంచనా వేయంది.",
        "age": "వయసు",
        "workclass": "పని తరగతి",
        "education": "విద్య",
        "occupation": "వృత్తి",
        "hours": "వారంకి గంటలు",
        "predict": "జీతం అంచనా",
        "upload_csv": "CSV అప్లోడ్ చేయండి",
        "download": "ఫలితాలను డౌన్లోడ్ చేయండి",
        "predicted_income": "💰 అంచన జీతం"
    }
}

# Language Selection
language = st.sidebar.selectbox("🌐 Select Language", list(LANGUAGES.keys()))
lang = LANGUAGES[language]

# UI
st.set_page_config(layout="wide")
st.title(lang["title"])
st.write(lang["intro"])
st.markdown("---")

# Manual Form Input
st.subheader("🔎 Manual Input")
age = st.number_input(lang["age"], 18, 100)
workclass = st.selectbox(lang["workclass"], ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education = st.selectbox(lang["education"], ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"])
occupation = st.selectbox(lang["occupation"], ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
hours_per_week = st.slider(lang["hours"], 1, 100, 40)

if st.button(lang["predict"]):
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
    st.success(f"{lang['predicted_income']}: **{prediction}**")

    # Save input_df globally to use in charts
    globals()['input_df'] = input_df

    # 🎨 Charts
    st.subheader("📊 Visual Insights from Input Data")

    fig_pie = px.pie(input_df, names='workclass', title="Workclass Distribution", color_discrete_sequence=['#2196F3'])
    fig_pie.update_traces(textinfo='percent+label', pull=[0.05])
    fig_pie.update_layout(template='plotly_white')
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_bar = px.bar(input_df, x='occupation', y='hours_per_week', color='occupation', title="Occupation vs Hours per Week", color_discrete_sequence=['#2196F3'])
    fig_bar.update_layout(xaxis_tickangle=-45, template='plotly_white')
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_hist = px.histogram(input_df, x='age', nbins=10, title="Age Distribution of Employees", color_discrete_sequence=['#2196F3'])
    fig_hist.update_layout(template='plotly_white')
    st.plotly_chart(fig_hist, use_container_width=True)

# CSV Upload for Bulk Predictions (Now at Bottom)
st.markdown("---")
st.subheader(f"📁 {lang['upload_csv']}")
uploaded_file = st.file_uploader("", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    predictions = model.predict(df_encoded)
    df['Prediction'] = predictions

    st.success("✅ Predictions done for uploaded data")
    st.dataframe(df)

    # 📊 Charts
    st.subheader("📊 Charts from Bulk Data")

    if 'workclass' in df.columns:
        fig_pie = px.pie(df, names='workclass', title="Workclass Distribution", color_discrete_sequence=['#2196F3'])
        fig_pie.update_layout(template='plotly_white')
        st.plotly_chart(fig_pie, use_container_width=True)

    if 'occupation' in df.columns and 'hours_per_week' in df.columns:
        fig_bar = px.bar(df, x='occupation', y='hours_per_week', title="Occupation vs Hours per Week", color='occupation', color_discrete_sequence=['#2196F3'])
        fig_bar.update_layout(template='plotly_white', xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    if 'age' in df.columns:
        fig_hist = px.histogram(df, x='age', nbins=10, title="Age Distribution", color_discrete_sequence=['#2196F3'])
        fig_hist.update_layout(template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)

    # Download predictions
    output = BytesIO()
    df.to_csv(output, index=False)
    st.download_button(label=lang["download"], data=output.getvalue(), file_name="predicted_results.csv", mime="text/csv")import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from io import BytesIO

# Load model and columns
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# Multilingual options
LANGUAGES = {
    "English": {
        "title": "💼 Employee Salary Classification App",
        "intro": "Predict whether an employee earns >50K or ≤50K based on input features.",
        "age": "Age",
        "workclass": "Workclass",
        "education": "Education",
        "occupation": "Occupation",
        "hours": "Hours per Week",
        "predict": "Predict Salary",
        "upload_csv": "Upload CSV for Bulk Predictions",
        "download": "Download Results",
        "predicted_income": "💰 Predicted Income"
    },
    "Hindi": {
        "title": "💼 कर्मचारी वेतन वर्गीकरण ऐप",
        "intro": "इनपुट सुविधाओं के आधार पर अनुमान लगाएं कि कर्मचारी की आय >50K या ≤50K है।",
        "age": "आयु",
        "workclass": "कार्य वर्ग",
        "education": "शिक्षा",
        "occupation": "पेशा",
        "hours": "प्रति सप्ताह घंटे",
        "predict": "वेतन की भविष्यवाणी करें",
        "upload_csv": "CSV अपलोड करें",
        "download": "परिणाम डाउनलोड करें",
        "predicted_income": "💰 अनुमानित आय"
    },
    "Telugu": {
        "title": "💼 ఉద్యోగి జీతం వర్గీకరణ యాప్",
        "intro": "ఇన్పుట్ ఫీచర్ల ఆధారంగా ఉద్యోగి ఆదాయం >50K లేదా ≤50K అని అంచనా వేయండి.",
        "age": "వయసు",
        "workclass": "పని తరగతి",
        "education": "విద్య",
        "occupation": "వృత్తి",
        "hours": "వారానికి గంటలు",
        "predict": "జీతం అంచనా",
        "upload_csv": "CSV అప్లోడ్ చేయండి",
        "download": "ఫలితాలను డౌన్‌లోడ్ చేయండి",
        "predicted_income": "💰 అంచనా జీతం"
    }
}

# Language Selection
language = st.sidebar.selectbox("🌐 Select Language", list(LANGUAGES.keys()))
lang = LANGUAGES[language]

# UI
st.set_page_config(layout="wide")
st.title(lang["title"])
st.write(lang["intro"])
st.markdown("---")

# CSV Upload for Bulk Predictions
uploaded_file = st.file_uploader(f"📁 {lang['upload_csv']}", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    predictions = model.predict(df_encoded)
    df['Prediction'] = predictions

    st.success("✅ Predictions done for uploaded data")
    st.dataframe(df)

    # 📊 Charts
    st.subheader("📊 Charts from Bulk Data")

    if 'workclass' in df.columns:
        fig_pie = px.pie(df, names='workclass', title="Workclass Distribution", color_discrete_sequence=['#2196F3'])
        fig_pie.update_layout(template='plotly_white')
        st.plotly_chart(fig_pie, use_container_width=True)

    if 'occupation' in df.columns and 'hours_per_week' in df.columns:
        fig_bar = px.bar(df, x='occupation', y='hours_per_week', title="Occupation vs Hours per Week", color='occupation', color_discrete_sequence=['#2196F3'])
        fig_bar.update_layout(template='plotly_white', xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    if 'age' in df.columns:
        fig_hist = px.histogram(df, x='age', nbins=10, title="Age Distribution", color_discrete_sequence=['#2196F3'])
        fig_hist.update_layout(template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)

    # Download predictions
    output = BytesIO()
    df.to_csv(output, index=False)
    st.download_button(label=lang["download"], data=output.getvalue(), file_name="predicted_results.csv", mime="text/csv")

else:
    # Manual Form Input
    st.subheader("🔎 Manual Input")
    age = st.number_input(lang["age"], 18, 100)
    workclass = st.selectbox(lang["workclass"], ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
    education = st.selectbox(lang["education"], ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school"])
    occupation = st.selectbox(lang["occupation"], ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    hours_per_week = st.slider(lang["hours"], 1, 100, 40)

    if st.button(lang["predict"]):
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
        st.success(f"{lang['predicted_income']}: **{prediction}**")

        # Save input_df globally to use in charts
        globals()['input_df'] = input_df

        # 🎨 Charts
        st.subheader("📊 Visual Insights from Input Data")

        fig_pie = px.pie(input_df, names='workclass', title="Workclass Distribution", color_discrete_sequence=['#2196F3'])
        fig_pie.update_traces(textinfo='percent+label', pull=[0.05])
        fig_pie.update_layout(template='plotly_white')
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_bar = px.bar(input_df, x='occupation', y='hours_per_week', color='occupation', title="Occupation vs Hours per Week", color_discrete_sequence=['#2196F3'])
        fig_bar.update_layout(xaxis_tickangle=-45, template='plotly_white')
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_hist = px.histogram(input_df, x='age', nbins=10, title="Age Distribution of Employees", color_discrete_sequence=['#2196F3'])
        fig_hist.update_layout(template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)
