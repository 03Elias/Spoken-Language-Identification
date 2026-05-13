"""
train_linear.py

Trains a logistic regression classifier on wav2vec 2.0 embeddings.
Reports validation accuracy and saves the trained model + label encoder.

Usage:
    python train_linear.py
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


OUTPUT_DIR = "../outputs"


def main():
    # Load embeddings + labels
    X_train = np.load(os.path.join(OUTPUT_DIR, "embeddings_train.npy"))
    y_train_str = np.load(os.path.join(OUTPUT_DIR, "labels_train.npy"))

    X_val = np.load(os.path.join(OUTPUT_DIR, "embeddings_val.npy"))
    y_val_str = np.load(os.path.join(OUTPUT_DIR, "labels_val.npy"))

    print(f"Train: {X_train.shape}, {len(y_train_str)} labels")
    print(f"Val:   {X_val.shape}, {len(y_val_str)} labels")

    # Encode labels (strings → ints for sklearn, kept consistent across splits)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_val = label_encoder.transform(y_val_str)

    print(f"Classes: {list(label_encoder.classes_)}")

    # Train logistic regression
    print("\nTraining logistic regression...")
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate on train + val
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")

    print("\nValidation classification report:")
    print(classification_report(
        y_val,
        val_preds,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    # Save classifier + label encoder
    clf_path = os.path.join(OUTPUT_DIR, "linear_classifier.pkl")
    enc_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

    joblib.dump(clf, clf_path)
    joblib.dump(label_encoder, enc_path)

    print(f"\nSaved:")
    print(f"  {clf_path}")
    print(f"  {enc_path}")


if __name__ == "__main__":
    main()