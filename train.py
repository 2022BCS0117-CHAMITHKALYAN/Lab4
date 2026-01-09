import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# -------------------------------
# 2. Train-Test Split (80/20)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 4. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("EXP-01: Linear Regression (Baseline)")
print("MSE:", mse)
print("R2 Score:", r2)

# -------------------------------
# 5. Save Outputs
# -------------------------------
os.makedirs("output", exist_ok=True)

joblib.dump(model, "output/model.pkl")

results = {
    "Experiment": "EXP-01",
    "Model": "Linear Regression",
    "MSE": mse,
    "R2_Score": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)
