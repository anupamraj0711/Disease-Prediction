import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# 1. Load & Clean Dataset
# ----------------------------
data = pd.read_csv("Training.csv")

# Drop unnamed/extra columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data.dropna()

X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# ----------------------------
# 2. Feature Selection (Top 95)
# ----------------------------
selector = SelectKBest(chi2, k=95)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# ----------------------------
# 3. Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 4. Train Random Forest
# ----------------------------
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

# ----------------------------
# 5. Evaluation
# ----------------------------
y_pred = rf.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# print("\n✅ Model Accuracy:", model_accuracy)
# # print("\nClassification Report:\n", classification_report(y_test, y_pred))
# # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ----------------------------
# 6. Test with Patient Symptoms
# ----------------------------

patient_symptoms = {sym: 0 for sym in selected_features}
# for sym in ["belly_pain", "chest_pain", "hip_joint_pain", "knee_pain", "back_pain"]:
for sym in ["continuous_sneezing","fever"]:
    if sym in patient_symptoms:
        patient_symptoms[sym] = 1

patient_input = pd.DataFrame([patient_symptoms])[selected_features].values
predicted_disease = rf.predict(patient_input)[0]

print("\nPredicted Disease:", predicted_disease)
print("Accuracy of Model:", model_accuracy)
