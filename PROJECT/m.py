import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Training.csv")
data = data.loc[:, ~data.columns.str.contains('^Unnamed')].dropna()
X, y = data.drop("prognosis", axis=1), data["prognosis"]

selector = SelectKBest(chi2, k=95)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

X_train, X_test, y_train, y_test = train_test_split(
    pd.DataFrame(X_new, columns=selected_features), y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

print(f"\nModel Accuracy: {accuracy_score(y_test, rf.predict(X_test))}")

while True:
    print("\nEnter up to 5 symptoms one by one ('done' to finish, 'exit' to quit):")
    patient = {sym: 0 for sym in selected_features}

    for i in range(5):
        s = input(f"Symptom {i+1}: ").strip()

        if s.lower() in ["done", "exit"]:
            break
        key = s.lower().replace(" ", "_")

        if key in patient:
            patient[key] = 1

    patient_input = pd.DataFrame([patient], columns=selected_features)
print("Predicted Disease:", rf.predict(patient_input)[0])
