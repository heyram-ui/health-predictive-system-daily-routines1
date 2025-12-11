import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# -------- FEATURE COLUMNS --------
FEATURE_COLUMNS = [
    "Gender", "Age", "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level", "BMI Category",
    "Heart Rate", "Daily Steps"
]

joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")
print("feature_columns.pkl created.")

# -------- SCALER --------
# This is a dummy scaler (you would normally fit this on real training data)
scaler = StandardScaler()
# Example: fit dummy scaler on zeros for correct dimensions
import pandas as pd
import numpy as np
dummy_data = pd.DataFrame(np.zeros((10, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
scaler.fit(dummy_data)
joblib.dump(scaler, "scaler.pkl")
print("scaler.pkl created.")

# -------- LABEL ENCODERS --------
label_encoders = {
    "Gender": LabelEncoder().fit(["Male", "Female"]),
    "BMI Category": LabelEncoder().fit(["Underweight", "Normal", "Overweight", "Obese"]),
    "Sleep Disorder": LabelEncoder().fit(["No Disorder", "Insomnia", "Sleep Apnea"])
}
joblib.dump(label_encoders, "label_encoder.pkl")
print("label_encoder.pkl created.")

# -------- LOGISTIC REGRESSION MODEL --------
# Dummy model (normally trained on real data)
X_dummy = scaler.transform(dummy_data)
y_dummy = label_encoders["Sleep Disorder"].transform(
    ["No Disorder", "Insomnia", "Sleep Apnea", "No Disorder", "Insomnia",
     "Sleep Apnea", "No Disorder", "Insomnia", "Sleep Apnea", "No Disorder"]
)
logistic_model = LogisticRegression()
logistic_model.fit(X_dummy, y_dummy)
joblib.dump(logistic_model, "logistic_model.pkl")
print("logistic_model.pkl created.")
