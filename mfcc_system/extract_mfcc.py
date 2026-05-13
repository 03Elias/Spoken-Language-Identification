import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

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

def run_extraction(split_name):
    csv_path = Path(f"data/metadata/{split_name}.csv")
    if not csv_path.exists():
        print(f"Metadata not found at {csv_path}. Run Elmira's script first!")
        return

    df = pd.read_csv(csv_path)
    features = []
    labels = []

    print(f"Extracting MFCCs for {split_name}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            feat = extract_features(row['path'])
            features.append(feat)
            labels.append(row['language'])
        except Exception as e:
            print(f"Error processing {row['path']}: {e}")

    np.save(f"mfcc_system/X_{split_name}.npy", np.array(features))
    np.save(f"mfcc_system/y_{split_name}.npy", np.array(labels))

if __name__ == "__main__":
    run_extraction("train")
    run_extraction("val")
    run_extraction("test")