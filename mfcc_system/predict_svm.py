import pandas as pd
import numpy as np
import joblib
import time

def predict():
    # Load test data and model
    X_test = np.load("mfcc_system/X_test.npy")
    y_test = np.load("mfcc_system/y_test.npy")
    model = joblib.load("mfcc_system/svm_model.pkl")
    scaler = joblib.load("mfcc_system/scaler.pkl")

    X_test_scaled = scaler.transform(X_test)

    # Measure inference time for the evaluation section
    start_time = time.time()
    predictions = model.predict(X_test_scaled)
    end_time = time.time()

    print(f"Inference time for {len(y_test)} samples: {end_time - start_time:.4f} seconds")

    # Create CSV for Elias
    results = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": predictions
    })
    results.to_csv("mfcc_system/svm_predictions.csv", index=False)
    print("Predictions saved to mfcc_system/svm_predictions.csv")

if __name__ == "__main__":
    predict()