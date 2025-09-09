from fastapi import FastAPI
from api.schemas import BiometricInput, StressResponse
from src.predict import predict_stress

app = FastAPI(title="BioAware Stress Detection API")

@app.post("/predict", response_model=StressResponse)
def predict(data: BiometricInput):
    prob = predict_stress(data.dict())
    return StressResponse(stress_probability=prob)
