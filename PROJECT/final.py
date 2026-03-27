import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Training.csv")

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data.dropna()

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

y_pred = rf.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)
print(" Model Accuracy:", model_accuracy)

valid_symptoms = list(selected_features)

while True:
    print("\nEnter symptoms (type 'done' when finished, max 5 symptoms).")
    print("Type 'exit' to quit the program.")
    input_symptoms = []
    
    while len(input_symptoms) < 5:
        symptom = input(f"Enter symptom {len(input_symptoms)+1}: ").strip().lower().replace(" ", "_")
        
        if symptom == "exit":
            print("Exiting patient test...")
            exit()
        
        if symptom == "done":
            break
        
        if symptom in valid_symptoms:
            input_symptoms.append(symptom)
        else:
            print(f"Warning: '{symptom}' not recognized, ignored.")

    patient_symptoms = {sym: 0 for sym in selected_features}

    for sym in input_symptoms:
        patient_symptoms[sym] = 1

    patient_input = pd.DataFrame([patient_symptoms])[selected_features].values
    predicted_disease = rf.predict(patient_input)[0]
    
    print("\nEntered Symptoms:", input_symptoms)
    print("Predicted Disease:", predicted_disease)
    print("Accuracy of Model:", model_accuracy)