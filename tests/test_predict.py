from src.predict import predict_stress

def test_prediction():
    sample = {"eda_mean": 0.3, "ecg_mean": 0.2, "acc_mean": 1.1}
    prob = predict_stress(sample)
    assert 0 <= prob <= 1
