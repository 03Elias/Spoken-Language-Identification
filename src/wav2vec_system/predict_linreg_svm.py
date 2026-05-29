"""Predict test labels with a trained wav2vec classifier."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent.parent / "outputs"


def embedding_path(split: str, layer: int) -> Path:
    candidate = OUTPUT_DIR / f"embeddings_layer_{layer}_{split}.npy"
    if candidate.exists():
        return candidate
    if layer == 12:
        return OUTPUT_DIR / f"embeddings_{split}.npy"
    return candidate


def _feature_time(layer: int) -> float | None:
    path = OUTPUT_DIR / "feature_times_wav2vec_test.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame = frame[frame["feature_layer"].astype(str) == str(layer)]
    if "feature_extraction_time_seconds" not in frame.columns or frame.empty:
        return None
    return float(frame["feature_extraction_time_seconds"].mean())


def predict(layer: int = 12, classifier: str = "logreg") -> Path:
    X_test = np.load(embedding_path("test", layer))
    y_test_str = np.load(OUTPUT_DIR / "labels_test.npy")
    clip_ids = np.load(OUTPUT_DIR / "clip_ids_test.npy")

    model_path = OUTPUT_DIR / f"wav2vec_layer_{layer}_{classifier}.pkl"
    enc_path = OUTPUT_DIR / f"wav2vec_layer_{layer}_{classifier}_label_encoder.pkl"
    if layer == 12 and classifier == "logreg" and not model_path.exists():
        model_path = OUTPUT_DIR / "linear_classifier.pkl"
        enc_path = OUTPUT_DIR / "label_encoder.pkl"

    clf = joblib.load(model_path)
    label_encoder = joblib.load(enc_path)

    start = time.perf_counter()
    preds = clf.predict(X_test)
    classifier_time = time.perf_counter() - start
    classifier_time_per_clip = classifier_time / len(X_test) if len(X_test) else 0.0
    pred_labels = label_encoder.inverse_transform(preds)

    feature_time_per_clip = _feature_time(layer)
    end_to_end = (
        feature_time_per_clip + classifier_time_per_clip
        if feature_time_per_clip is not None
        else None
    )

    test_acc = accuracy_score(y_test_str, pred_labels)
    print(f"Test accuracy: {test_acc:.4f}")
    print(
        "Classifier inference time after feature extraction: "
        f"{classifier_time:.4f}s total"
    )

    experiment_id = f"wav2vec_layer_{layer}_{classifier}"
    frame = pd.DataFrame(
        {
            "experiment_id": experiment_id,
            "feature_type": "wav2vec",
            "feature_layer": layer,
            "classifier": classifier,
            "clip_id": clip_ids,
            "true_label": y_test_str,
            "predicted_label": pred_labels,
            "feature_extraction_time_seconds": feature_time_per_clip,
            "classifier_inference_time_seconds": classifier_time_per_clip,
            "end_to_end_inference_time_seconds": end_to_end,
        }
    )

    csv_path = OUTPUT_DIR / f"predictions_{experiment_id}.csv"
    frame.to_csv(csv_path, index=False)
    if layer == 12 and classifier == "logreg":
        frame.to_csv(OUTPUT_DIR / "predictions_wav2vec.csv", index=False)
    print(f"Saved predictions to {csv_path}")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a wav2vec classifier.")
    parser.add_argument("--layer", type=int, default=12, help="wav2vec hidden-state layer.")
    parser.add_argument(
        "--classifier",
        default="logreg",
        choices=["logreg", "svm_rbf"],
        help="Classifier to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict(layer=args.layer, classifier=args.classifier)


if __name__ == "__main__":
    main()
