# train.py
import os, json
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
import joblib

# create output directory
os.makedirs("output", exist_ok=True)

# load dataset
X, y = load_wine(return_X_y=True)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)

# compute metrics
f1 = float(f1_score(y_test, pred, average="weighted"))
mse = float(mean_squared_error(y_test, pred))

# save model
joblib.dump(model, "output/model.pkl")

# save metrics.json (THIS IS USED BY DEPLOY JOB)
metrics = {
    "f1": round(f1, 4),
    "mse": round(mse, 4)
}
with open("output/metrics.json", "w") as f:
    json.dump(metrics, f)

# save results.json (for Job Summary)
results = {
    "name": "BOLLI CHAMITH KALYAN",
    "roll_no": "2022BCS0117",
    "metrics": metrics
}
with open("output/results.json", "w") as f:
    json.dump(results, f)

print("Training complete.")
print("Metrics:", metrics)
