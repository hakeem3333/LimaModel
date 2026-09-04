"""
LimaModel FastAPI — Stress Detection Service

Accepts biometric fields collected by the LimaHealth wearable pipeline and
returns a stress probability from the trained WESAD model bundle.

Feature mapping (app field → WESAD feature):
  gsr       → eda_mean  (galvanic skin response ≈ electrodermal activity)
  skin_temp → temp_mean (wrist skin temperature)
  heart_rate → bvp_mean (heart rate correlates with BVP amplitude)
  movement  → acc_mean  (accelerometer magnitude)

Single-reading std/min/max are approximated with neutral defaults.
"""

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = "models/wesad_stress_model.pkl"

try:
    model_bundle = joblib.load(MODEL_PATH)
    clf    = model_bundle["model"]
    scaler = model_bundle["scaler"]
    MODEL_LOADED = True
except Exception as e:
    print(f"[LimaModel] WARNING: Could not load model — {e}")
    clf = scaler = None
    MODEL_LOADED = False

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BiometricInput(BaseModel):
    """
    Biometric fields available from the LimaHealth wearable pipeline.
    All values are averages over a recent time window (e.g. last 60 seconds).
    """
    heart_rate:   float = Field(..., ge=0,   description="Average heart rate (bpm)")
    gsr:          float = Field(..., ge=0,   description="Galvanic skin response (EDA proxy)")
    skin_temp:    float = Field(..., ge=0,   description="Wrist skin temperature (°C)")
    movement:     float = Field(..., ge=0,   description="Accelerometer magnitude mean")

    # Optional enrichments — used when available
    gsr_std:      Optional[float] = Field(default=0.0, ge=0)
    skin_temp_std:Optional[float] = Field(default=0.0, ge=0)
    heart_rate_std:Optional[float]= Field(default=0.0, ge=0)
    movement_std: Optional[float] = Field(default=0.0, ge=0)
    hrv:          Optional[float] = Field(default=0.0, ge=0, description="Heart-rate variability (SDNN)")


class StressResponse(BaseModel):
    prediction:        int   # 1 = stress, 0 = no stress
    stress_probability:float  # [0.0, 1.0]
    risk_level:        str   # CRITICAL | HIGH | MODERATE | LOW


def probability_to_risk(prob: float) -> str:
    if prob >= 0.80: return "CRITICAL"
    if prob >= 0.60: return "HIGH"
    if prob >= 0.40: return "MODERATE"
    return "LOW"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_feature_vector(data: BiometricInput) -> np.ndarray:
    """
    Construct the 15-element feature vector expected by the WESAD model.

    Order matches training CSV columns:
      eda_mean, eda_std, eda_min, eda_max,
      temp_mean, temp_std, temp_min, temp_max,
      bvp_mean, bvp_std, bvp_min, bvp_max, bvp_hrv,
      acc_mean, acc_std
    """
    eda_mean  = data.gsr
    eda_std   = data.gsr_std or 0.0
    eda_min   = max(0.0, eda_mean - eda_std)
    eda_max   = eda_mean + eda_std

    temp_mean = data.skin_temp
    temp_std  = data.skin_temp_std or 0.0
    temp_min  = max(0.0, temp_mean - temp_std)
    temp_max  = temp_mean + temp_std

    # Normalise heart rate to a BVP-like amplitude (rough linear proxy)
    bvp_mean  = data.heart_rate / 100.0
    bvp_std   = (data.heart_rate_std or 0.0) / 100.0
    bvp_min   = max(0.0, bvp_mean - bvp_std)
    bvp_max   = bvp_mean + bvp_std
    bvp_hrv   = data.hrv or 0.0

    acc_mean  = data.movement
    acc_std   = data.movement_std or 0.0

    return np.array([[
        eda_mean, eda_std, eda_min, eda_max,
        temp_mean, temp_std, temp_min, temp_max,
        bvp_mean, bvp_std, bvp_min, bvp_max, bvp_hrv,
        acc_mean, acc_std,
    ]])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LimaModel — Stress Detection API",
    description="Predicts student stress probability from wearable biometric data.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}


@app.post("/predict", response_model=StressResponse)
def predict(data: BiometricInput):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="ML model is not loaded")

    X = build_feature_vector(data)
    X_scaled = scaler.transform(X)

    pred  = int(clf.predict(X_scaled)[0])
    prob  = float(clf.predict_proba(X_scaled)[0][1])
    risk  = probability_to_risk(prob)

    return StressResponse(
        prediction=pred,
        stress_probability=round(prob, 4),
        risk_level=risk,
    )
