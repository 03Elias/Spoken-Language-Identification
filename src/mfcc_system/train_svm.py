import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

def train():
    print("Loading extracted features...")
    X_train = np.load(BASE_DIR / "X_train.npy")
    y_train = np.load(BASE_DIR / "y_train.npy")

    # Scaling is MANDATORY for SVMs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM with RBF Kernel
    print("Training SVM (this might take a moment)...")
    model = SVC(kernel='rbf', probability=True, C=1.0)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler for prediction/evaluation
    joblib.dump(model, BASE_DIR / "svm_model.pkl")
    joblib.dump(scaler, BASE_DIR / "scaler.pkl")
    print("Success: Model and Scaler saved.")

if __name__ == "__main__":
    train()