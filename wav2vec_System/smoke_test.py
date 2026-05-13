"""
smoke_test.py

Generates a tiny fake dataset (3 languages × 5 clips per split) and runs the
full wav2vec pipeline end-to-end. Used to verify everything works before
plugging in real CommonVoice data.

Usage:
    python smoke_test.py
"""

import os
import subprocess
import numpy as np
import pandas as pd
import soundfile as sf


SMOKE_DIR = "smoke_test_data"
AUDIO_DIR = os.path.join(SMOKE_DIR, "audio")
SPLITS_DIR = os.path.join(SMOKE_DIR, "splits")
SAMPLE_RATE = 16000
DURATION_SEC = 2.0
LANGUAGES = ["english", "swedish", "german"]
CLIPS_PER_LANG_PER_SPLIT = 5


def generate_fake_audio(path, language_idx):
    """Generate 2 seconds of noise with a language-specific frequency bias.
    This is fake — just enough variation that the classifier has *something*
    to latch onto for the smoke test."""
    t = np.linspace(0, DURATION_SEC, int(SAMPLE_RATE * DURATION_SEC))
    base_freq = 200 + language_idx * 150  # Different base freq per language
    signal = (
        0.3 * np.sin(2 * np.pi * base_freq * t)
        + 0.1 * np.random.randn(len(t))
    )
    sf.write(path, signal.astype(np.float32), SAMPLE_RATE)


def create_split_csv(split_name):
    """Create a CSV for one split."""
    rows = []
    for lang_idx, lang in enumerate(LANGUAGES):
        for i in range(CLIPS_PER_LANG_PER_SPLIT):
            clip_id = f"{split_name}_{lang}_{i:03d}"
            file_path = os.path.abspath(
                os.path.join(AUDIO_DIR, f"{clip_id}.wav")
            )
            generate_fake_audio(file_path, lang_idx)
            rows.append({
                "clip_id": clip_id,
                "file_path": file_path,
                "language": lang,
                "split": split_name,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    print("=" * 60)
    print("SMOKE TEST — wav2vec language ID pipeline")
    print("=" * 60)

    # Setup
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs("../outputs", exist_ok=True)

    # Generate fake data
    print("\n[1/4] Generating fake audio + split CSVs...")
    for split in ["train", "val", "test"]:
        csv_path = create_split_csv(split)
        print(f"  Created {csv_path}")

    # We need to point the extract script at our smoke-test splits dir
    # Easiest: monkeypatch via env var, OR temporarily symlink.
    # Simpler: just import and call directly so we override the path.
    print("\n[2/4] Extracting embeddings...")
    import sys
    sys.path.insert(0, ".")

    # Override paths in the extract module
    import extract_wav2vec
    extract_wav2vec.SPLITS_DIR = SPLITS_DIR

    for split in ["train", "val", "test"]:
        print(f"\n--- Extracting {split} ---")
        sys.argv = ["extract_wav2vec.py", "--split", split]
        extract_wav2vec.main()

    # Train
    print("\n[3/4] Training classifier...")
    import train_linear
    train_linear.main()

    # Predict
    print("\n[4/4] Predicting on test set...")
    import predict_linear
    predict_linear.main()

    # Final checks
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)

    csv_path = "../outputs/predictions_wav2vec.csv"
    df = pd.read_csv(csv_path)
    expected_cols = {"clip_id", "true_label", "predicted_label"}

    checks = [
        ("Output CSV exists", os.path.exists(csv_path)),
        ("Correct columns", set(df.columns) == expected_cols),
        ("Right number of rows", len(df) == len(LANGUAGES) * CLIPS_PER_LANG_PER_SPLIT),
        ("No NaN predictions", not df["predicted_label"].isna().any()),
        ("Labels are strings", df["true_label"].dtype == object),
    ]

    for desc, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")

    all_passed = all(p for _, p in checks)
    print("\n" + ("ALL CHECKS PASSED ✓" if all_passed else "SOME CHECKS FAILED ✗"))


if __name__ == "__main__":
    main()