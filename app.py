import streamlit as st
import pandas as pd
import plotly.express as px

from prediction_functions import (
    predict_risk,
    generate_suggestions,
    get_feature_importance
)

st.set_page_config(page_title="EduShield AI", layout="wide")

st.title("🎓 EduShield AI")
st.markdown("### Student Risk Prediction & Intervention System")

# ---------------- INPUT SECTION ----------------
st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_time = st.slider("Study Time (1-4)", 1, 4, 2)
    absences = st.number_input("Absences", 0, 100, 5)

with col2:
    failures = st.number_input("Past Failures", 0, 5, 0)
    health = st.slider("Health (1-5)", 1, 5, 3)

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Student"):

    risk, confidence = predict_risk(
        study_time, absences, failures, health
    )

    st.subheader("📊 Results")
    st.success(f"Risk Level: {risk}")
    st.info(f"Confidence: {confidence}%")

    # Suggestions
    st.subheader("🧠 Suggestions")
    suggestions = generate_suggestions(
        study_time, absences, failures, health
    )
    for s in suggestions:
        st.write(f"- {s}")

    # Feature Importance
    st.subheader("📈 Feature Importance")
    importance = get_feature_importance()

    if importance:
        df_imp = pd.DataFrame({
            "Feature": list(importance.keys()),
            "Importance": list(importance.values())
        })

        fig = px.bar(df_imp, x="Feature", y="Importance",
                     title="Feature Importance")
        st.plotly_chart(fig)
    else:
        st.warning("Feature importance not available.")

# ---------------- CSV UPLOAD ----------------
st.subheader("📂 Bulk Student Analysis")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview:", df.head())

    try:
        risks = []
        confidences = []

        for _, row in df.iterrows():
            r, c = predict_risk(
                row["study_time"],
                row["absences"],
                row["failures"],
                row["health"]
            )
            risks.append(r)
            confidences.append(c)

        df["Risk"] = risks
        df["Confidence"] = confidences

        st.subheader("📊 Results")
        st.dataframe(df)

        # Visualization
        fig = px.histogram(df, x="Risk", title="Risk Distribution")
        st.plotly_chart(fig)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results",
            csv,
            "results.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("CSV format incorrect. Required columns: study_time, absences, failures, health")
