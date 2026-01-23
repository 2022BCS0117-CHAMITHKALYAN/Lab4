# train.py
import os, json
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
import joblib
import numpy as np

os.makedirs("output", exist_ok=True)

# load example dataset
X, y = load_wine(return_X_y=True)

# simple binary / multi-class example - keep as-is
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# compute metrics (example)
f1 = float(f1_score(y_test, pred, average='weighted'))
mse = float(mean_squared_error(y_test, pred))

metrics = {"f1": round(f1, 4), "mse": round(mse, 4)}
joblib.dump(model, "output/model.pkl")

# results.json - human readable summary (this is used in job summary)
results = {
    "name": "BOLLI CHAMITH KALYAN",
    "roll_no": "2022BCS0117",
    "metrics": metrics
}

with open("output/metrics.json", "w") as f:
    json.dump(metrics, f)

with open("output/results.json", "w") as f:
    json.dump(results, f)

print("Saved model and metrics to output/")
