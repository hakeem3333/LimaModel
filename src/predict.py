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
