"""Extract frozen wav2vec 2.0 embeddings for selected transformer layers."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor


MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
EMBED_DIM = 768
DEFAULT_LAYERS = [12]

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
SPLITS_DIR = PROJECT_ROOT / "src" / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_model(device: torch.device) -> tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


def resolve_audio_path(raw_path: str | Path) -> Path:
    audio_path = Path(raw_path)
    if audio_path.exists():
        return audio_path

    project_path = PROJECT_ROOT / audio_path
    if project_path.exists():
        return project_path

    src_path = PROJECT_ROOT / "src" / audio_path
    if src_path.exists():
        return src_path

    return audio_path


def extract_layer_embeddings(
    audio_path: str | Path,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
    layers: list[int],
) -> dict[int, np.ndarray]:
    """Return mean-pooled embeddings for each requested transformer layer."""
    audio, _ = librosa.load(resolve_audio_path(audio_path), sr=SAMPLE_RATE)
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    embeddings: dict[int, np.ndarray] = {}
    for layer in layers:
        if layer < 0 or layer >= len(hidden_states):
            raise ValueError(
                f"Layer {layer} is unavailable; valid hidden_state indexes are "
                f"0..{len(hidden_states) - 1}"
            )
        embeddings[layer] = hidden_states[layer].mean(dim=1).squeeze().cpu().numpy()

    return embeddings


def _file_and_clip_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    if "file_path" in df.columns:
        file_col = "file_path"
    elif "path" in df.columns:
        file_col = "path"
    else:
        raise ValueError("Expected a 'file_path' or 'path' column.")
    clip_col = "clip_id" if "clip_id" in df.columns else None
    return file_col, clip_col


def _save_outputs(
    split: str,
    layers: list[int],
    layer_embeddings: dict[int, np.ndarray],
    labels: list[str],
    clip_ids: list[str],
    timings: list[dict[str, object]],
) -> None:
    labels_array = np.asarray(labels)
    clip_ids_array = np.asarray(clip_ids)

    for layer in layers:
        emb_path = OUTPUT_DIR / f"embeddings_layer_{layer}_{split}.npy"
        np.save(emb_path, layer_embeddings[layer])
        if layer == 12:
            np.save(OUTPUT_DIR / f"embeddings_{split}.npy", layer_embeddings[layer])

    np.save(OUTPUT_DIR / f"labels_{split}.npy", labels_array)
    np.save(OUTPUT_DIR / f"clip_ids_{split}.npy", clip_ids_array)
    pd.DataFrame(timings).to_csv(
        OUTPUT_DIR / f"feature_times_wav2vec_{split}.csv",
        index=False,
    )


def run_extraction(split: str, layers: list[int]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = SPLITS_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} clips from {csv_path}")

    file_col, clip_col = _file_and_clip_columns(df)
    print(f"Loading {MODEL_NAME}...")
    processor, model = load_model(device)

    layer_embeddings = {
        layer: np.zeros((len(df), EMBED_DIM), dtype=np.float32) for layer in layers
    }
    labels: list[str] = []
    clip_ids: list[str] = []
    timings: list[dict[str, object]] = []
    failed: list[str] = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
        file_path = row[file_col]
        cid = str(row[clip_col]) if clip_col else Path(str(file_path)).stem
        try:
            start = time.perf_counter()
            embeddings = extract_layer_embeddings(file_path, processor, model, device, layers)
            elapsed = time.perf_counter() - start
            for layer, embedding in embeddings.items():
                layer_embeddings[layer][i] = embedding
                timings.append(
                    {
                        "clip_id": cid,
                        "feature_type": "wav2vec",
                        "feature_layer": layer,
                        "feature_extraction_time_seconds": elapsed,
                    }
                )
            labels.append(str(row["language"]) if "language" in df.columns else "")
            clip_ids.append(cid)
        except Exception as error:
            print(f"\nFailed on {cid}: {error}")
            failed.append(cid)
            labels.append(str(row["language"]) if "language" in df.columns else "")
            clip_ids.append(cid)

    _save_outputs(split, layers, layer_embeddings, labels, clip_ids, timings)
    if failed:
        print(f"Warning: {len(failed)} clips failed extraction: {failed[:5]}...")

    for layer in layers:
        assert layer_embeddings[layer].shape == (len(df), EMBED_DIM)
        assert not np.isnan(layer_embeddings[layer]).any()
        print(
            f"Saved layer {layer}: "
            f"{OUTPUT_DIR / f'embeddings_layer_{layer}_{split}.npy'} "
            f"shape={layer_embeddings[layer].shape}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract wav2vec layer embeddings.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Which split to extract.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=DEFAULT_LAYERS,
        help="Hidden-state indexes to mean-pool. For base wav2vec, 12 is final.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_extraction(args.split, sorted(set(args.layers)))


if __name__ == "__main__":
    main()
