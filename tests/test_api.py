from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_predict():
    response = client.post("/predict", json={"eda_mean":0.2, "ecg_mean":0.1, "acc_mean":1.5})
    assert response.status_code == 200
    assert "stress_probability" in response.json()
