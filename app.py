import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="EduShield AI", layout="wide")

# Load model
model = joblib.load("model.pkl")

# SHAP explainer
explainer = shap.Explainer(model)

risk_map = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🎓 EduShield AI</h1>
    <p style='text-align: center;'>Student Risk Prediction & Explainability System</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📥 Enter Student Details")

name = st.sidebar.text_input("Student Name")

study_time = st.sidebar.slider("Study Time", 1, 4, 2)
absences = st.sidebar.number_input("Absences", 0, 100, 5)
failures = st.sidebar.number_input("Failures", 0, 5, 0)
health = st.sidebar.slider("Health", 1, 5, 3)

analyze = st.sidebar.button("🔍 Analyze Student")

# ---------------- MAIN ----------------
col1, col2 = st.columns(2)

if analyze:
    input_data = np.array([[study_time, absences, failures, health]])

    prediction = model.predict(input_data)[0]
    confidence = max(model.predict_proba(input_data)[0]) * 100

    student_name = name if name else "Student"

    with col1:
        st.markdown("### 📊 Prediction Result")

        st.markdown(f"""
            <div style='padding:20px;
                        border-radius:10px;
                        background-color:#1e1e1e;
                        text-align:center'>
                <h2 style='color:#00ffcc'>{student_name}: {risk_map[prediction]}</h2>
                <p>Confidence: {round(confidence,2)}%</p>
            </div>
        """, unsafe_allow_html=True)

    # ---------------- SUGGESTIONS ----------------
    with col2:
        st.markdown("### 🧠 Suggestions")

        tips = []

        if study_time <= 1:
            tips.append("Increase study time.")
        if absences > 10:
            tips.append("Reduce absences.")
        if failures > 0:
            tips.append("Focus on weak subjects.")
        if health < 3:
            tips.append("Improve health habits.")

        if not tips:
            tips.append("Keep up the good performance!")

        for tip in tips:
            st.markdown(f"""
                <div style='padding:10px;
                            margin:5px;
                            border-radius:8px;
                            background-color:#262730'>
                    {tip}
                </div>
            """, unsafe_allow_html=True)
if analyze:
    input_data = np.array([[study_time, absences, failures, health]])

    prediction = model.predict(input_data)[0]
    confidence = max(model.predict_proba(input_data)[0]) * 100

    # ---------------- SHAP ----------------
    st.markdown("### 🔍 Why this prediction? (Explainability)")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    feature_names = ["study_time", "absences", "failures", "health"]

    values = shap_values.values

    if len(values.shape) == 3:
        values = values[0][prediction]
    else:
        values = values[0]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": values
    })

    shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

    st.dataframe(shap_df, use_container_width=True)

# ---------------- CSV SECTION ----------------
st.markdown("---")
st.markdown("## 📂 Bulk Student Analysis")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.dataframe(df, use_container_width=True)

    try:
        risks = []
        confidences = []

        for _, row in df.iterrows():
            input_data = np.array([[row["study_time"], row["absences"],
                                    row["failures"], row["health"]]])

            pred = model.predict(input_data)[0]
            conf = max(model.predict_proba(input_data)[0]) * 100

            risks.append(risk_map[pred])
            confidences.append(round(conf, 2))

        df["Risk"] = risks
        df["Confidence"] = confidences

        # High Risk Filter
        st.markdown("### 🚨 High Risk Students")
        high_risk = df[df["Risk"] == "High Risk"]

        if not high_risk.empty:
            st.dataframe(high_risk, use_container_width=True)
        else:
            st.success("No high-risk students 🎉")

        st.markdown("### 📊 Full Results")
        st.dataframe(df, use_container_width=True)

    except:
        st.error("CSV format must include: name, study_time, absences, failures, health")
