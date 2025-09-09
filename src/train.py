import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from src.preprocess import preprocess_features

def train_model(dataset_csv="data/processed/wesad_features.csv", save_path="models/wesad_stress_model.pkl"):
    df = pd.read_csv(dataset_csv)

    X = df.drop("label", axis=1)
    y = df["label"].apply(lambda x: 1 if x == 2 else 0)  # stress=1, non-stress=0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(preprocess_features(X_train, fit=True), y_train)

    preds = clf.predict(preprocess_features(X_test))

    print(classification_report(y_test, preds))

    joblib.dump(clf, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
