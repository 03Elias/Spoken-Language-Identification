"""
predict_linear.py

Loads the trained classifier and predicts on the test set.
Writes predictions_wav2vec.csv in the agreed format for the evaluation module.

Usage:
    python predict_linear.py
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


OUTPUT_DIR = "../outputs"


def main():
    # Load test embeddings + labels + clip_ids
    X_test = np.load(os.path.join(OUTPUT_DIR, "embeddings_test.npy"))
    y_test_str = np.load(os.path.join(OUTPUT_DIR, "labels_test.npy"))
    clip_ids = np.load(os.path.join(OUTPUT_DIR, "clip_ids_test.npy"))

    print(f"Test: {X_test.shape}, {len(y_test_str)} labels")

    # Load classifier + label encoder
    clf = joblib.load(os.path.join(OUTPUT_DIR, "linear_classifier.pkl"))
    label_encoder = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    # Predict (time it for the inference-cost comparison)
    start = time.time()
    preds = clf.predict(X_test)
    elapsed = time.time() - start

    pred_labels = label_encoder.inverse_transform(preds)

    # Quick test accuracy (for sanity — Person 4 will do proper evaluation)
    test_acc = accuracy_score(y_test_str, pred_labels)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Inference time (classifier only): {elapsed:.3f}s for {len(X_test)} clips")
    print(f"  → {elapsed / len(X_test) * 1000:.2f} ms per clip")

    # Save predictions in the agreed format
    df = pd.DataFrame({
        "clip_id": clip_ids,
        "true_label": y_test_str,
        "predicted_label": pred_labels,
    })

    csv_path = os.path.join(OUTPUT_DIR, "predictions_wav2vec.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")
    print(df.head())


if __name__ == "__main__":
    main()