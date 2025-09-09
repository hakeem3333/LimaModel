# LimaHealth ML Service

This repository contains the machine learning service for **LimaHealth**, an AI-driven student wellness and stress monitoring platform.  
It provides a Python-based pipeline to train, save, and deploy predictive models for stress detection using wearable sensor data.

## ✨ Features
- Data preprocessing & feature extraction for biometric signals
- Model training with scikit-learn (e.g., Random Forest, Logistic Regression)
- Model persistence (`.pkl`) with joblib
- REST API built with FastAPI to serve real-time predictions
- Dockerized for easy deployment (AWS, Render, Railway, etc.)
- Designed to integrate with the LimaHealth Node.js backend

```
## 📂 Project Structure

LimaModel/
├── data/ # Local datasets (not deployed)
├── models/ # Trained models
├── src/ # Training, preprocessing, prediction code
├── api/ # FastAPI app for serving predictions
├── tests/ # Unit tests
├── Pipfile # Pipenv dependencies
└── Dockerfile # Containerization for deployment
```


## 🚀 Quick Start
```bash
# Install dependencies
pipenv install

# Train model
pipenv run python src/train.py

# Run API
pipenv run uvicorn api.main:app --reload

🧪 API Usage

After running the API:

POST /predict
Content-Type: application/json

{
  "heart_rate": 80,
  "eda": 0.45,
  "temperature": 36.5,
  "acc_x": 0.01,
  "acc_y": -0.02,
  "acc_z": 0.98
}


Response:

{
  "stress_score": 0.73,
  "label": "stressed"
}

📊 Dataset

The model can be trained on academic datasets like WESAD (Wearable Stress and Affect Detection)
 until real LimaHealth data is available.

📜 License

For research and educational purposes. Dataset usage subject to original terms (WESAD is CC BY-NC-SA).
