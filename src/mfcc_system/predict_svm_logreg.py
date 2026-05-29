"""Predict test labels with a trained MFCC classifier."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent


def _cache_path(prefix: str, split: str, feature_type: str) -> Path:
    candidate = BASE_DIR / f"{prefix}_{split}_{feature_type}.npy"
    if candidate.exists():
        return candidate
    if feature_type == "baseline" and prefix in {"X", "y"}:
        return BASE_DIR / f"{prefix}_{split}.npy"
    return candidate


def _clip_ids(split: str, feature_type: str, n_rows: int) -> np.ndarray:
    path = _cache_path("clip_ids", split, feature_type)
    if path.exists():
        return np.load(path)
    return np.asarray([f"{split}_{i:04d}" for i in range(n_rows)])


def _feature_time(split: str, feature_type: str) -> float | None:
    path = BASE_DIR / f"feature_times_{split}_{feature_type}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "feature_extraction_time_seconds" not in frame.columns or frame.empty:
        return None
    return float(frame["feature_extraction_time_seconds"].mean())


def predict(feature_type: str = "baseline", classifier: str = "svm_rbf") -> Path:
    X_test = np.load(_cache_path("X", "test", feature_type))
    y_test = np.load(_cache_path("y", "test", feature_type))
    clip_ids = _clip_ids("test", feature_type, len(y_test))

    model_path = BASE_DIR / f"mfcc_{feature_type}_{classifier}.pkl"
    if not model_path.exists() and feature_type == "baseline" and classifier == "svm_rbf":
        model_path = BASE_DIR / "svm_model.pkl"
    model = joblib.load(model_path)

    start = time.perf_counter()
    predictions = model.predict(X_test)
    classifier_time = time.perf_counter() - start
    classifier_time_per_clip = classifier_time / len(y_test) if len(y_test) else 0.0
    feature_time_per_clip = _feature_time("test", feature_type)
    end_to_end = (
        feature_time_per_clip + classifier_time_per_clip
        if feature_time_per_clip is not None
        else None
    )

    test_acc = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {test_acc:.4f}")
    print(
        "Classifier inference time after feature extraction: "
        f"{classifier_time:.4f}s total"
    )

    experiment_id = f"mfcc_{feature_type}_{classifier}"
    results = pd.DataFrame(
        {
            "experiment_id": experiment_id,
            "feature_type": f"mfcc_{feature_type}",
            "feature_layer": "",
            "classifier": classifier,
            "clip_id": clip_ids,
            "true_label": y_test,
            "predicted_label": predictions,
            "feature_extraction_time_seconds": feature_time_per_clip,
            "classifier_inference_time_seconds": classifier_time_per_clip,
            "end_to_end_inference_time_seconds": end_to_end,
        }
    )

    output_path = BASE_DIR / f"predictions_{experiment_id}.csv"
    results.to_csv(output_path, index=False)
    if feature_type == "baseline" and classifier == "svm_rbf":
        results.to_csv(BASE_DIR / "svm_predictions.csv", index=False)
    print(f"Predictions saved to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with an MFCC classifier.")
    parser.add_argument(
        "--feature-type",
        default="baseline",
        choices=["baseline", "temporal"],
        help="MFCC feature cache to use.",
    )
    parser.add_argument(
        "--classifier",
        default="svm_rbf",
        choices=["logreg", "svm_rbf"],
        help="Classifier to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict(feature_type=args.feature_type, classifier=args.classifier)


if __name__ == "__main__":
    main()
