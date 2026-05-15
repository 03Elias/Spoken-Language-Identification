import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

def predict():
    # Load test data and model
    X_test = np.load(BASE_DIR / "X_test.npy")
    y_test = np.load(BASE_DIR / "y_test.npy")
    model = joblib.load(BASE_DIR / "svm_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")

    X_test_scaled = scaler.transform(X_test)

    # Measure inference time for the evaluation section
    start_time = time.time()
    predictions = model.predict(X_test_scaled)
    end_time = time.time()

    inference_time_seconds = end_time - start_time
    inference_time_per_sample_seconds = (
        inference_time_seconds / len(y_test) if len(y_test) else 0.0
    )

    print(f"Inference time for {len(y_test)} samples: {inference_time_seconds:.4f} seconds")

    # Create CSV for Elias
    results = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": predictions,
        "inference_time_seconds": inference_time_seconds,
        "inference_time_per_sample_seconds": inference_time_per_sample_seconds,
    })
    results.to_csv(BASE_DIR / "svm_predictions.csv", index=False)
    print(f"Predictions saved to {BASE_DIR / 'svm_predictions.csv'}")

if __name__ == "__main__":
    predict()