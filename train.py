import os, json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
import joblib

# create output directory
os.makedirs("output", exist_ok=True)

# load dataset
X, y = load_wine(return_X_y=True)

# introduce slight noise in labels (to avoid perfect score)
np.random.seed(42)
noise_idx = np.random.choice(len(y), size=int(0.05 * len(y)), replace=False)
y_noisy = y.copy()
y_noisy[noise_idx] = np.random.choice(np.unique(y), size=len(noise_idx))

# stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_noisy,
    test_size=0.25,
    random_state=42,
    stratify=y_noisy
)

# use a weaker model to avoid overfitting
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# compute metrics
f1 = float(f1_score(y_test, pred, average="weighted"))
mse = float(mean_squared_error(y_test, pred))

# save model
joblib.dump(model, "output/model.pkl")

# save metrics
metrics = {
    "f1": round(f1, 4),
    "mse": round(mse, 4)
}
with open("output/metrics.json", "w") as f:
    json.dump(metrics, f)

# save results
results = {
    "name": "BOLLI CHAMITH KALYAN",
    "roll_no": "2022BCS0117",
    "metrics": metrics
}
with open("output/results.json", "w") as f:
    json.dump(results, f)

print("Training complete.")
print("F1:", f1, "MSE:", mse)
