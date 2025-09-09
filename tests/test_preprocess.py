import pandas as pd
from src.preprocess import preprocess_features

def test_preprocess():
    df = pd.DataFrame({"eda_mean": [0.2], "ecg_mean": [0.1], "acc_mean": [1.5]})
    result = preprocess_features(df, fit=True)
    assert result.shape == (1, 3)
