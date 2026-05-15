import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata"
OUTPUT_ROOT = BASE_DIR

def extract_features(audio_path):
    # 1. Load 16kHz audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # 2. Pre-emphasis (High-frequency boost)
    y = librosa.effects.preemphasis(y)
    
    # 3. Extract 13 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 4. Aggregate (Mean & Std) to get a 26-dim vector
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.hstack((mean, std))


def resolve_audio_path(raw_path):
    path = Path(raw_path)

    if path.exists():
        return path

    src_path = PROJECT_ROOT / path
    if src_path.exists():
        return src_path

    return path

def run_extraction(split_name):
    csv_path = METADATA_ROOT / f"{split_name}.csv"
    if not csv_path.exists():
        print(f"Metadata not found at {csv_path}. Run Elmira's script first!")
        return

    df = pd.read_csv(csv_path)
    features = []
    labels = []

    print(f"Extracting MFCCs for {split_name}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            audio_path = resolve_audio_path(row["path"])
            feat = extract_features(audio_path)
            features.append(feat)
            labels.append(row['language'])
        except Exception as e:
            print(f"Error processing {row['path']}: {e}")

    np.save(OUTPUT_ROOT / f"X_{split_name}.npy", np.array(features))
    np.save(OUTPUT_ROOT / f"y_{split_name}.npy", np.array(labels))

if __name__ == "__main__":
    run_extraction("train")
    run_extraction("val")
    run_extraction("test")