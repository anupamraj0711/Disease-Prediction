import pandas as pd

# Load dataset
data = pd.read_csv("Training.csv")

# Drop any completely empty or unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Drop rows with NaN (if any)
data = data.dropna()

# Separate features and labels
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=95)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Selected features:", selected_features)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



import numpy as np

patient_symptoms = {sym: 0 for sym in selected_features}
for sym in ["belly_pain", "chest_pain", "hip_joint_pain", "knee_pain", "back_pain"]:
    if sym in patient_symptoms:
        patient_symptoms[sym] = 1




patient_input = pd.DataFrame([patient_symptoms])[selected_features]
print("Predicted Disease:", rf.predict(patient_input.values)[0])
