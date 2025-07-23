# Everything else remains same up to this line...

# Manual Form Input
st.subheader("🔎 Manual Input")
age = st.number_input(lang["age"], 18, 100)
workclass = st.selectbox(lang["workclass"], [...])
education = st.selectbox(lang["education"], [...])
occupation = st.selectbox(lang["occupation"], [...])
hours_per_week = st.slider(lang["hours"], 1, 100, 40)

predict_clicked = st.button(lang["predict"])  # SINGLE button

if predict_clicked:
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

# --- CSV Upload for Bulk Predictions ---
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

    st.subheader("📊 Charts from Bulk Data")
    # same charts for bulk data...

    output = BytesIO()
    df.to_csv(output, index=False)
    st.download_button(label=lang["download"], data=output.getvalue(), file_name="predicted_results.csv", mime="text/csv")
