from pydantic import BaseModel

class BiometricInput(BaseModel):
    eda_mean: float
    ecg_mean: float
    acc_mean: float

class StressResponse(BaseModel):
    stress_probability: float
