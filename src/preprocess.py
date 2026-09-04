# src/preprocess.py
import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_hrv(bvp_signal, fs=64):
    """
    Compute a simple HRV feature from BVP:
    - Detect peaks
    - Compute RR intervals
    - Return std of RR intervals (SDNN)
    """
    bvp_signal = np.ravel(bvp_signal)  # ensure 1D
    peaks, _ = find_peaks(bvp_signal, distance=int(fs * 0.6))  # min distance ~60 BPM
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        return np.std(rr_intervals)
    return 0.0


def preprocess_features(
    raw_data_dir="data/raw/WESAD",
    output_csv="data/processed/wesad_features.csv",
    window_size=60,   # seconds per window
    step_size=30      # overlap (seconds)
):
    rows = []

    for subject in os.listdir(raw_data_dir):
        subject_path = os.path.join(raw_data_dir, subject, f"{subject}.pkl")
        if not os.path.isfile(subject_path):
            continue

        print(f"Processing {subject_path}...")
        with open(subject_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        # Use only wrist signals
        if "wrist" not in data["signal"]:
            print(f"⚠️ No wrist signals in {subject}, skipping...")
            continue
        wrist = data["signal"]["wrist"]

        # Sampling rates for wrist signals
        fs = {
            "ACC": 32,     # accelerometer (x,y,z)
            "EDA": 4,      # electrodermal activity
            "TEMP": 4,     # temperature
            "BVP": 64      # blood volume pulse
        }

        # Labels are 1 per second
        labels = data["label"]
        total_time = len(labels)
        skipped = 0  # count skipped windows

        for start_sec in range(0, total_time - window_size, step_size):
            end_sec = start_sec + window_size
            label_window = labels[start_sec:end_sec]

            if len(label_window) < window_size:
                continue

            # Binary label: 1 = stress (2 in WESAD), 0 otherwise
            label = 1 if np.mean(label_window == 2) > 0.5 else 0

            # Extract signal windows
            eda_seg = wrist["EDA"][start_sec*fs["EDA"]:end_sec*fs["EDA"]]
            temp_seg = wrist["TEMP"][start_sec*fs["TEMP"]:end_sec*fs["TEMP"]]
            bvp_seg = wrist["BVP"][start_sec*fs["BVP"]:end_sec*fs["BVP"]]
            acc_seg = wrist["ACC"][start_sec*fs["ACC"]:end_sec*fs["ACC"], :]

            # Skip if any segment is empty
            if (len(eda_seg) == 0 or len(temp_seg) == 0 or
                len(bvp_seg) == 0 or acc_seg.shape[0] == 0):
                skipped += 1
                continue

            acc_mag = np.linalg.norm(acc_seg, axis=1)

            # Feature extraction
            features = {
                "eda_mean": np.mean(eda_seg),
                "eda_std": np.std(eda_seg),
                "eda_min": np.min(eda_seg),
                "eda_max": np.max(eda_seg),

                "temp_mean": np.mean(temp_seg),
                "temp_std": np.std(temp_seg),
                "temp_min": np.min(temp_seg),
                "temp_max": np.max(temp_seg),

                "bvp_mean": np.mean(bvp_seg),
                "bvp_std": np.std(bvp_seg),
                "bvp_min": np.min(bvp_seg),
                "bvp_max": np.max(bvp_seg),
                "bvp_hrv": extract_hrv(bvp_seg, fs=fs["BVP"]),

                "acc_mean": np.mean(acc_mag),
                "acc_std": np.std(acc_mag),
            }

            rows.append({**features, "label": label})

        print(f"✅ Finished {subject}: {len(rows)} windows kept so far, {skipped} skipped in this subject")

    print("Total windows processed:", len(rows))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved processed dataset with {df.shape[1]-1} features and {df.shape[0]} samples to {output_csv}")


if __name__ == "__main__":
    preprocess_features()
