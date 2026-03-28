import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")

model = joblib.load("model.pkl")

risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# ----------- CUSTOM CSS -----------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.card {
    background: rgba(255,255,255,0.7);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}
.title {
    font-size: 28px;
    font-weight: bold;
}
.badge-low {color: green;}
.badge-med {color: orange;}
.badge-high {color: red;}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<div class='title'>🎓 EduShield AI Dashboard</div>", unsafe_allow_html=True)
st.markdown("Student Monitoring & Risk Analysis System")

st.markdown("---")

# -------- SIDEBAR --------
st.sidebar.header("Input")

name = st.sidebar.text_input("Student Name")
study_time = st.sidebar.slider("Study Time", 1, 4, 2)
absences = st.sidebar.number_input("Absences", 0, 100, 5)
failures = st.sidebar.number_input("Failures", 0, 5, 0)
health = st.sidebar.slider("Health", 1, 5, 3)

analyze = st.sidebar.button("Analyze")

# -------- MAIN GRID --------
left, right = st.columns([3,1])


# -------- LEFT PANEL --------
with left:
    st.markdown("### 📊 Student Overview")

    if analyze:
        data = np.array([[study_time, absences, failures, health]])
        pred = model.predict(data)[0]
        conf = max(model.predict_proba(data)[0]) * 100

        # -------- RESULT --------
        label = risk_map[pred]
        color_class = "badge-low" if pred==0 else "badge-med" if pred==1 else "badge-high"

        st.markdown(f"""
        <div class='card'>
            <h3>{name if name else "Student"}</h3>
            <h2 class='{color_class}'>{label}</h2>
            <p>Confidence: {round(conf,2)}%</p>
        </div>
        """, unsafe_allow_html=True)

        # -------- SUGGESTIONS --------
        st.markdown("### 🧠 Suggestions")

        tips = []
        if study_time <= 1: tips.append("Increase study time")
        if absences > 10: tips.append("Reduce absences")
        if failures > 0: tips.append("Focus on weak subjects")
        if health < 3: tips.append("Improve health")

        if not tips:
            tips.append("Keep performing well!")

        for t in tips:
            st.markdown(f"<div class='card'>{t}</div>", unsafe_allow_html=True)

        # -------- ANALYSIS REPORT (✅ FIXED POSITION) --------
        st.markdown("### 📈 Analysis Report")

        report_col1, report_col2, report_col3 = st.columns(3)

        with report_col1:
            st.metric("Study Time", study_time)

        with report_col2:
            st.metric("Absences", absences)

        with report_col3:
            st.metric("Failures", failures)

        chart_df = pd.DataFrame({
            "Feature": ["Study Time", "Absences", "Failures", "Health"],
            "Value": [study_time, absences, failures, health]
        })

        st.bar_chart(chart_df.set_index("Feature"))
# -------- SIMPLE BAR CHART --------
chart_df = pd.DataFrame({
    "Feature": ["Study Time", "Absences", "Failures", "Health"],
    "Value": [study_time, absences, failures, health]
})

st.bar_chart(chart_df.set_index("Feature"))

        label = risk_map[pred]
        color_class = "badge-low" if pred==0 else "badge-med" if pred==1 else "badge-high"

        st.markdown(f"""
        <div class='card'>
            <h3>{name if name else "Student"}</h3>
            <h2 class='{color_class}'>{label}</h2>
            <p>Confidence: {round(conf,2)}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🧠 Suggestions")

        tips = []
        if study_time <= 1: tips.append("Increase study time")
        if absences > 10: tips.append("Reduce absences")
        if failures > 0: tips.append("Focus on weak subjects")
        if health < 3: tips.append("Improve health")

        if not tips: tips.append("Keep performing well!")

        for t in tips:
            st.markdown(f"<div class='card'>{t}</div>", unsafe_allow_html=True)

# -------- RIGHT PANEL --------
with right:
    st.markdown("### ⚙️ Actions")

    st.markdown("<div class='card'>📊 View Reports</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>🤖 Auto Suggestions</div>", unsafe_allow_html=True)

# -------- CSV SECTION --------
st.markdown("---")
st.markdown("### 📂 Manage Students")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    risks = []
    for _, row in df.iterrows():
        data = np.array([[row["study_time"], row["absences"],
                          row["failures"], row["health"]]])
        pred = model.predict(data)[0]
        risks.append(risk_map[pred])

    df["Risk"] = risks

    tabs = st.tabs(["All", "High Risk", "Medium", "Low"])

    with tabs[0]:
        st.dataframe(df)

    with tabs[1]:
        st.dataframe(df[df["Risk"] == "High Risk"])

    with tabs[2]:
        st.dataframe(df[df["Risk"] == "Medium Risk"])

    with tabs[3]:
        st.dataframe(df[df["Risk"] == "Low Risk"])
