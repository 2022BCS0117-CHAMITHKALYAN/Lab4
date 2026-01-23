# app.py
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = "model.pkl"  # when packaged into Docker we will copy this to root

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

class InputFeatures(BaseModel):
    features: list

@app.get("/predict")
def predict_get():
    # simple demo response required by lab
    return {"name": "BOLLI CHAMITH KALYAN", "roll_no": "2022BCS0117", "wine_quality": 5}

@app.post("/predict")
def predict_post(inp: InputFeatures):
    if model is None:
        return {"error": "model not loaded"}
    x = np.array(inp.features).reshape(1, -1)
    pred = model.predict(x)
    return {"wine_quality": int(pred[0])}
