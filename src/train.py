import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Paths
DATASET_CSV = "data/processed/wesad_features.csv"
SAVE_PATH = "models/wesad_stress_model.pkl"

def train_model(dataset_csv=DATASET_CSV, save_path=SAVE_PATH):
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"❌ Dataset not found at {dataset_csv}")

    # Load dataset
    df = pd.read_csv(dataset_csv)

    # Features and labels
    X = df.drop("label", axis=1)
    y = df["label"].apply(lambda x: 1 if x == 2 else 0)  # stress=1, non-stress=0

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    preds = clf.predict(X_test_scaled)
    print(classification_report(y_test, preds))

    # Save both model + scaler in one bundle
    joblib.dump({"model": clf, "scaler": scaler}, save_path)
    print(f"✅ Model + Scaler saved to {save_path}")

if __name__ == "__main__":
    train_model()
