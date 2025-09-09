import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def preprocess_features(df: pd.DataFrame, fit: bool = False):
    """
    Preprocess features before training or prediction.
    - Handles scaling
    - Ensures consistent feature order
    """
    features = ["eda_mean", "ecg_mean", "acc_mean"]

    X = df[features].values

    if fit:
        return scaler.fit_transform(X)
    else:
        return scaler.transform(X)

def create_features_from_raw(eda, ecg, acc, window_size=700):
    """
    Convert raw signals into windowed features.
    """
    eda_mean = np.mean(eda.reshape(-1, window_size), axis=1)
    ecg_mean = np.mean(ecg.reshape(-1, window_size), axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)
    acc_mean = np.mean(acc_mag.reshape(-1, window_size), axis=1)

    return pd.DataFrame({
        "eda_mean": eda_mean,
        "ecg_mean": ecg_mean,
        "acc_mean": acc_mean
    })
