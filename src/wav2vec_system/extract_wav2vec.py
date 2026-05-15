"""
extract_wav2vec.py

Extracts frozen wav2vec 2.0 embeddings for a given data split.
Saves embeddings, labels, and clip_ids as .npy files for downstream use.

Usage:
    python extract_wav2vec.py --split train
    python extract_wav2vec.py --split val
    python extract_wav2vec.py --split test
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm


MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
EMBED_DIM = 768

# Paths — adjust to match the group's agreed folder structure
SPLITS_DIR = "data/metadata"
OUTPUT_DIR = "../outputs"


def load_model(device):
    """Load frozen wav2vec 2.0 base model."""
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)
    # Freeze all weights (defensive — eval() already disables dropout/BN updates,
    # but this guarantees no gradients are tracked)
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


def extract_embedding(audio_path, processor, model, device):
    """Extract a single 768-dim mean-pooled embedding from one audio file."""
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    # outputs.last_hidden_state shape: (1, time, 768)
    # Mean-pool across time → (768,)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Which data split to extract embeddings for",
    )
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load split CSV
    csv_path = os.path.join(SPLITS_DIR, f"{args.split}.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} clips from {csv_path}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    processor, model = load_model(device)

    # Determine which columns to use for file path and clip id
    if "file_path" in df.columns:
        file_col = "file_path"
    elif "path" in df.columns:
        file_col = "path"
    else:
        raise ValueError(f"No file path column found in {csv_path}; expected 'file_path' or 'path'.")

    clip_col = "clip_id" if "clip_id" in df.columns else None

    # Extract embeddings
    embeddings = np.zeros((len(df), EMBED_DIM), dtype=np.float32)
    labels = []
    clip_ids = []
    failed = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {args.split}"):
        file_path = row[file_col]
        # derive clip id if missing
        if clip_col:
            cid = row[clip_col]
        else:
            cid = Path(file_path).stem

        try:
            emb = extract_embedding(file_path, processor, model, device)
            embeddings[i] = emb
            labels.append(row["language"] if "language" in df.columns else None)
            clip_ids.append(cid)
        except Exception as e:
            print(f"\nFailed on {cid}: {e}")
            failed.append(cid)
            labels.append(row["language"] if "language" in df.columns else None)
            clip_ids.append(cid)

    if failed:
        print(f"\nWarning: {len(failed)} clips failed extraction: {failed[:5]}...")

    # Save
    emb_path = os.path.join(OUTPUT_DIR, f"embeddings_{args.split}.npy")
    lbl_path = os.path.join(OUTPUT_DIR, f"labels_{args.split}.npy")
    id_path = os.path.join(OUTPUT_DIR, f"clip_ids_{args.split}.npy")

    np.save(emb_path, embeddings)
    np.save(lbl_path, np.array(labels))
    np.save(id_path, np.array(clip_ids))

    print(f"\nSaved:")
    print(f"  {emb_path}  shape={embeddings.shape}")
    print(f"  {lbl_path}  ({len(labels)} labels)")
    print(f"  {id_path}   ({len(clip_ids)} clip_ids)")

    # Sanity checks
    assert embeddings.shape == (len(df), EMBED_DIM), "Unexpected embedding shape"
    assert not np.isnan(embeddings).any(), "NaN values in embeddings!"
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()