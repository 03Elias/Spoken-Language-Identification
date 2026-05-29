"""Extract cached MFCC feature matrices for the SLI experiments.

Two feature variants are supported:

* baseline: 13 MFCCs with mean/std pooling -> 26 dimensions
* temporal: MFCC + delta + delta-delta with mean/std pooling -> 78 dimensions
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata"
OUTPUT_ROOT = BASE_DIR
SAMPLE_RATE = 16000
N_MFCC = 13
FEATURE_DIMS = {
    "baseline": 26,
    "temporal": 78,
}


def resolve_audio_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path

    return path


def pool_mean_std(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=1)
    std = np.std(features, axis=1)
    return np.hstack((mean, std)).astype(np.float32)


def extract_features(audio_path: str | Path, feature_type: str = "baseline") -> np.ndarray:
    """Extract one utterance-level MFCC feature vector."""
    if feature_type not in FEATURE_DIMS:
        raise ValueError(f"Unknown MFCC feature type: {feature_type}")

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = librosa.effects.preemphasis(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    if feature_type == "baseline":
        feature_stack = mfcc
    else:
        delta = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        feature_stack = np.vstack([mfcc, delta, delta_delta])

    pooled = pool_mean_std(feature_stack)
    expected_dim = FEATURE_DIMS[feature_type]
    if pooled.shape[0] != expected_dim:
        raise ValueError(
            f"{feature_type} MFCC vector has {pooled.shape[0]} dims; expected {expected_dim}"
        )
    return pooled


def _clip_id(row: pd.Series) -> str:
    if "clip_id" in row and pd.notna(row["clip_id"]):
        return str(row["clip_id"])
    return Path(str(row["path"])).stem


def _save_cache(
    split_name: str,
    feature_type: str,
    features: list[np.ndarray],
    labels: list[str],
    clip_ids: list[str],
    timings: list[dict[str, object]],
) -> None:
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    ids = np.asarray(clip_ids)

    np.save(OUTPUT_ROOT / f"X_{split_name}_{feature_type}.npy", X)
    np.save(OUTPUT_ROOT / f"y_{split_name}_{feature_type}.npy", y)
    np.save(OUTPUT_ROOT / f"clip_ids_{split_name}_{feature_type}.npy", ids)
    pd.DataFrame(timings).to_csv(
        OUTPUT_ROOT / f"feature_times_{split_name}_{feature_type}.csv",
        index=False,
    )

    # Preserve the original filenames for the baseline pipeline.
    if feature_type == "baseline":
        np.save(OUTPUT_ROOT / f"X_{split_name}.npy", X)
        np.save(OUTPUT_ROOT / f"y_{split_name}.npy", y)


def run_extraction(split_name: str, feature_type: str) -> None:
    csv_path = METADATA_ROOT / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'path' column")
    if "language" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'language' column")

    features: list[np.ndarray] = []
    labels: list[str] = []
    clip_ids: list[str] = []
    timings: list[dict[str, object]] = []
    failures: list[str] = []

    print(f"Extracting {feature_type} MFCCs for {split_name}...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{feature_type} {split_name}"):
        cid = _clip_id(row)
        try:
            audio_path = resolve_audio_path(row["path"])
            start = time.perf_counter()
            feat = extract_features(audio_path, feature_type=feature_type)
            elapsed = time.perf_counter() - start

            features.append(feat)
            labels.append(str(row["language"]))
            clip_ids.append(cid)
            timings.append(
                {
                    "clip_id": cid,
                    "feature_type": f"mfcc_{feature_type}",
                    "feature_extraction_time_seconds": elapsed,
                }
            )
        except Exception as error:
            failures.append(cid)
            print(f"Error processing {row['path']}: {error}")

    _save_cache(split_name, feature_type, features, labels, clip_ids, timings)
    print(
        f"Saved {len(features)} {feature_type} MFCC vectors for {split_name}; "
        f"failed={len(failures)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract baseline or temporal MFCC caches.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to extract.",
    )
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=["baseline", "temporal"],
        choices=sorted(FEATURE_DIMS),
        help="MFCC feature variants to extract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for feature_type in args.feature_types:
        for split_name in args.splits:
            run_extraction(split_name, feature_type)


if __name__ == "__main__":
    main()
