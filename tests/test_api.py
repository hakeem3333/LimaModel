from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_predict():
    # Example input: same order as training features
    features = [0.2, 0.1, 1.5]

    response = client.post("/predict", json={"features": features})
    
    assert response.status_code == 200
    data = response.json()
    assert "stress_probability" in data
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
