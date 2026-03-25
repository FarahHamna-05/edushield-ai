import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Risk labels
risk_map = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

# ---------------- PREDICTION ----------------
def predict_risk(study_time, absences, failures, health):
    input_data = np.array([[study_time, absences, failures, health]])
    
    prediction = model.predict(input_data)[0]
    probability = np.max(model.predict_proba(input_data)) * 100

    return risk_map[prediction], round(probability, 2)

# ---------------- SUGGESTIONS ----------------
def generate_suggestions(study_time, absences, failures, health):
    suggestions = []

    if study_time <= 1:
        suggestions.append("Increase daily study time.")

    if absences > 10:
        suggestions.append("Reduce absences to improve performance.")

    if failures > 0:
        suggestions.append("Focus on weak subjects and revise regularly.")

    if health < 3:
        suggestions.append("Maintain better health for improved concentration.")

    if not suggestions:
        suggestions.append("Keep up the good performance!")

    return suggestions

# ---------------- FEATURE IMPORTANCE (SIMPLE) ----------------
def get_feature_importance():
    try:
        importance = model.feature_importances_
        features = ["study_time", "absences", "failures", "health"]

        return dict(zip(features, importance))
    except:
        return {}
