"""Train classifiers on cached wav2vec 2.0 embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent.parent / "outputs"


def embedding_path(split: str, layer: int) -> Path:
    candidate = OUTPUT_DIR / f"embeddings_layer_{layer}_{split}.npy"
    if candidate.exists():
        return candidate
    if layer == 12:
        return OUTPUT_DIR / f"embeddings_{split}.npy"
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


def train(layer: int = 12, classifier: str = "logreg") -> Path:
    X_train = np.load(embedding_path("train", layer))
    y_train_str = np.load(OUTPUT_DIR / "labels_train.npy")
    X_val = np.load(embedding_path("val", layer))
    y_val_str = np.load(OUTPUT_DIR / "labels_val.npy")

    print(f"Train: {X_train.shape}, {len(y_train_str)} labels")
    print(f"Val:   {X_val.shape}, {len(y_val_str)} labels")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_val = label_encoder.transform(y_val_str)
    print(f"Classes: {list(label_encoder.classes_)}")

    clf = build_classifier(classifier)
    print(f"Training wav2vec layer {layer} + {classifier}...")
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    print(f"Train accuracy: {accuracy_score(y_train, train_preds):.4f}")
    print(f"Val accuracy:   {accuracy_score(y_val, val_preds):.4f}")
    print(
        classification_report(
            y_val,
            val_preds,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    model_path = OUTPUT_DIR / f"wav2vec_layer_{layer}_{classifier}.pkl"
    enc_path = OUTPUT_DIR / f"wav2vec_layer_{layer}_{classifier}_label_encoder.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(label_encoder, enc_path)

    # Preserve old artifact names for final-layer Logistic Regression.
    if layer == 12 and classifier == "logreg":
        joblib.dump(clf, OUTPUT_DIR / "linear_classifier.pkl")
        joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.pkl")

    print(f"Saved model: {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a wav2vec embedding classifier.")
    parser.add_argument("--layer", type=int, default=12, help="wav2vec hidden-state layer.")
    parser.add_argument(
        "--classifier",
        default="logreg",
        choices=["logreg", "svm_rbf"],
        help="Classifier to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(layer=args.layer, classifier=args.classifier)


if __name__ == "__main__":
    main()
