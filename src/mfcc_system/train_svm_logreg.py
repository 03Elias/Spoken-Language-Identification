"""Train the MFCC classifiers"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent


def cache_path(split: str, feature_type: str) -> Path:
    candidate = BASE_DIR / f"X_{split}_{feature_type}.npy"
    if candidate.exists():
        return candidate
    if feature_type == "baseline":
        return BASE_DIR / f"X_{split}.npy"
    return candidate


def label_path(split: str, feature_type: str) -> Path:
    candidate = BASE_DIR / f"y_{split}_{feature_type}.npy"
    if candidate.exists():
        return candidate
    if feature_type == "baseline":
        return BASE_DIR / f"y_{split}.npy"
    return candidate


def build_classifier(classifier: str) -> Pipeline:
    if classifier == "logreg":
        estimator = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )
    elif classifier == "svm_rbf":
        estimator = SVC(kernel="rbf", probability=True, C=1.0, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def train(feature_type: str = "baseline", classifier: str = "svm_rbf") -> Path:
    print(f"Loading {feature_type} MFCC features...")
    X_train = np.load(cache_path("train", feature_type))
    y_train = np.load(label_path("train", feature_type))

    model = build_classifier(classifier)
    print(f"Training MFCC {feature_type} + {classifier}...")
    model.fit(X_train, y_train)

    val_path = cache_path("val", feature_type)
    if val_path.exists():
        X_val = np.load(val_path)
        y_val = np.load(label_path("val", feature_type))
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"Validation accuracy: {val_acc:.4f}")

    model_path = BASE_DIR / f"mfcc_{feature_type}_{classifier}.pkl"
    joblib.dump(model, model_path)

    # Preserve old artifact names for baseline SVM.
    if feature_type == "baseline" and classifier == "svm_rbf":
        joblib.dump(model, BASE_DIR / "svm_model.pkl")

    print(f"Saved model: {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MFCC classifier.")
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
        help="Classifier to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(feature_type=args.feature_type, classifier=args.classifier)


if __name__ == "__main__":
    main()
