# ===============================
# EXP-02: Ridge Regression + Standardization
# ===============================

import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load dataset
# -------------------------------
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

# -------------------------------
# 2. Define features and target
# -------------------------------
X = data.drop("quality", axis=1)
y = data["quality"]

# -------------------------------
# 3. Standardization
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. Train-test split (80/20)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Train model
# -------------------------------
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("EXP-02: Ridge Regression + Standardization")
print("MSE:", mse)
print("R2 Score:", r2)

# -------------------------------
# 7. Save outputs
# -------------------------------
os.makedirs("output", exist_ok=True)

joblib.dump(model, "output/model.pkl")

results = {
    "Experiment": "EXP-02",
    "Model": "Ridge Regression",
    "Alpha": 1.0,
    "Preprocessing": "Standardization",
    "MSE": mse,
    "R2_Score": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)
