import joblib
import pandas as pd
from src.preprocess import preprocess_features

MODEL_PATH = "models/wesad_stress_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_stress(input_dict: dict) -> float:
    """
    Takes in a dictionary with eda_mean, ecg_mean, acc_mean
    Returns probability of stress
    """
    df = pd.DataFrame([input_dict])
    X = preprocess_features(df, fit=False)
    proba = model.predict_proba(X)[0][1]  # stress probability
    return float(proba)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model + scaler
MODEL_PATH = "models/wesad_stress_model.pkl"
model_bundle = joblib.load(MODEL_PATH)
clf = model_bundle["model"]
scaler = model_bundle["scaler"]

# FastAPI app
app = FastAPI(title="LimaModel API", description="Stress Detection from Wearable Data")

class Features(BaseModel):
    features: list[float]  # raw feature vector (same order as training CSV)

@app.post("/predict")
def predict(data: Features):
    # Convert input to numpy
    X = np.array(data.features).reshape(1, -1)

    # Apply scaler
    X_scaled = scaler.transform(X)

    # Predict
    pred = clf.predict(X_scaled)[0]
    prob = clf.predict_proba(X_scaled)[0][1]  # probability of stress

    return {
        "prediction": int(pred),  # 1 = stress, 0 = non-stress
        "stress_probability": float(prob)
    }

