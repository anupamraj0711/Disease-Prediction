import pandas as pd
import joblib
import numpy as np
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Smart Disease Prediction", page_icon="🩺", layout="wide")

DATA_PATH = "Training.csv"
MODEL_PATH = "disease_model.pkl"
FEATURE_PATH = "features.pkl"
RESULTS_PATH = "prediction_results.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data = data.dropna()
    return data

@st.cache_resource
def train_or_load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        rf = joblib.load(MODEL_PATH)
        selected_features = joblib.load(FEATURE_PATH)
    else:
        data = load_data()
        X = data.drop("prognosis", axis=1)
        y = data["prognosis"]

        selector = SelectKBest(chi2, k=105)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]

        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y, test_size=0.2, random_state=42, stratify=y
        )

        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train, y_train)

        # Save model and features
        joblib.dump(rf, MODEL_PATH)
        joblib.dump(selected_features, FEATURE_PATH)

    return rf, selected_features

rf, selected_features = train_or_load_model()
data = load_data()

# =======================
# 2️⃣ PAGE STYLING
# =======================
st.markdown("""
    <style>
    .main-title { text-align: center; color: #2E86C1; font-size: 40px; font-weight: bold; }
    .section-title { color: #1A5276; font-size: 25px; margin-top: 20px; }
    .stButton>button { background-color: #2E86C1; color: white; border-radius: 10px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #1F618D; color: white; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🩺 Smart Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# =======================
# 3️⃣ MODEL PERFORMANCE
# =======================
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

X_new = X[selected_features]
y_pred = rf.predict(X_new)
model_accuracy = accuracy_score(y, y_pred)

st.markdown("<h2 class='section-title'>📊 Model Performance Overview</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Model Accuracy", value=f"{model_accuracy*100:.2f}%")

with col2:
    st.write("### Feature Importance")
    importances = rf.feature_importances_
    feature_importance = pd.Series(importances, index=selected_features)
    top_features = feature_importance.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Symptoms")
    ax.set_title("Top 10 Most Important Symptoms")
    st.pyplot(fig)

st.markdown("---")

# =======================
# 4️⃣ SYMPTOM INPUT + PREDICTION
# =======================
st.markdown("<h2 class='section-title'>🧬 Disease Prediction</h2>", unsafe_allow_html=True)

valid_symptoms = list(selected_features)
input_symptoms = []
max_symptoms = 5

cols = st.columns(max_symptoms)
for i, c in enumerate(cols):
    with c:
        symptom = st.text_input(f"Symptom {i+1}", key=f"sym_{i}").strip().lower().replace(" ", "_")
        if symptom:
            # ✅ Smart correction: keep as is
            if symptom not in valid_symptoms:
                suggestion = get_close_matches(symptom, valid_symptoms, n=1, cutoff=0.7)
                if suggestion:
                    st.info(f"Did you mean **{suggestion[0]}**?")
                    symptom = suggestion[0]
                else:
                    st.warning(f"'{symptom}' not recognized and ignored.")
                    continue
            input_symptoms.append(symptom)

# Predict Disease
if st.button("🔍 Predict Disease"):
    if not input_symptoms:
        st.error("Please enter at least one valid symptom.")
    else:
        try:
            # Patient symptom vector
            patient_symptoms = {sym: 0 for sym in selected_features}
            for sym in input_symptoms:
                patient_symptoms[sym] = 1

            patient_input = pd.DataFrame([patient_symptoms])[selected_features].values

            predicted_disease = rf.predict(patient_input)[0]
            probs = rf.predict_proba(patient_input)[0]
            top3 = sorted(zip(rf.classes_, probs), key=lambda x: x[1], reverse=True)[:3]

            st.success(f"### ✅ Predicted Disease: **{predicted_disease}**")
            st.write(f"**Model Accuracy:** {model_accuracy*100:.2f}%")

            st.subheader("Top 3 Possible Diseases")
            for disease, prob in top3:
                st.write(f"➡️ {disease}: {prob*100:.2f}%")

            # Save prediction results
            result = pd.DataFrame({
                "Entered Symptoms": [", ".join(input_symptoms)],
                "Predicted Disease": [predicted_disease],
                "Accuracy": [model_accuracy]
            })

            if not os.path.exists(RESULTS_PATH):
                result.to_csv(RESULTS_PATH, index=False)
            else:
                result.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
            st.info("✅ Prediction saved successfully!")

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

st.markdown("---")

# =======================
# 5️⃣ FOOTER
# =======================
st.caption("""
**Developed by:** Anupam  
**Guided by:** Dr. Shweta R. Malwe & Mrs. Prachi Mehta
""")
