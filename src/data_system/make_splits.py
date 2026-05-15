import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from preprocess_audio import preprocess_file_with_librosa, check_audio_file


LANGUAGES = [
    "en",
    "sv",
    "de",
    "fr",
    "es",
    "ar",
    "zh",
    "ru",
    "ja",
    "fi",
]


SPLIT_FILES = {
    "train": "train.tsv",
    "val": "dev.tsv",
    "test": "test.tsv",
}


def read_tsv(tsv_path):
    """
    Read a Common Voice TSV file.
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing TSV file: {tsv_path}")

    return pd.read_csv(tsv_path, sep="\t")


def prepare_split(
    lang_code,
    split_name,
    tsv_filename,
    n_samples,
    raw_root,
    output_root,
):
    """
    Prepare one split for one language.
    """

    if n_samples == 0:
        return []

    lang_raw_dir = Path(raw_root) / lang_code
    clips_dir = lang_raw_dir / "clips"
    tsv_path = lang_raw_dir / tsv_filename

    if not clips_dir.exists():
        raise FileNotFoundError(f"Missing clips folder: {clips_dir}")

    df = read_tsv(tsv_path)

    if "path" not in df.columns:
        raise ValueError(f"'path' column not found in {tsv_path}")

    rows = []
    count = 0
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{lang_code} {split_name}"):
        if count >= n_samples:
            break

        clip_filename = row["path"]
        input_path = clips_dir / clip_filename

        if not input_path.exists():
            skipped += 1
            continue

        output_filename = f"{lang_code}_{split_name}_{count:04d}.wav"
        output_path = Path(output_root) / split_name / lang_code / output_filename

        try:
            saved_path = preprocess_file_with_librosa(
                input_path=input_path,
                output_path=output_path,
            )

            check_audio_file(saved_path)

            rows.append(
                {
                    "path": str(saved_path),
                    "language": lang_code,
                    "split": split_name,
                    "sentence": row.get("sentence", ""),
                    "original_path": str(input_path),
                    "original_sampling_rate": 16000,
                }
            )

            count += 1

        except Exception as error:
            skipped += 1
            print(f"Skipped {input_path}: {error}")

    if count < n_samples:
        print(
            f"WARNING: only prepared {count}/{n_samples} samples "
            f"for {lang_code} {split_name}. Skipped: {skipped}"
        )

    return rows


def prepare_language(
    lang_code,
    raw_root,
    output_root,
    n_train,
    n_val,
    n_test,
):
    """
    Prepare train, validation and test splits for one language.
    """
    all_rows = []

    split_settings = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }

    for split_name, n_samples in split_settings.items():
        rows = prepare_split(
            lang_code=lang_code,
            split_name=split_name,
            tsv_filename=SPLIT_FILES[split_name],
            n_samples=n_samples,
            raw_root=raw_root,
            output_root=output_root,
        )
        all_rows.extend(rows)

    return all_rows


def save_metadata(metadata, metadata_root):
    """
    Save train.csv, val.csv, test.csv and all_metadata.csv.
    """
    metadata_root = Path(metadata_root)
    metadata_root.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(metadata_root / "all_metadata.csv", index=False)

    for split in ["train", "val", "test"]:
        if "split" in metadata.columns:
            split_df = metadata[metadata["split"] == split]
        else:
            split_df = pd.DataFrame()

        split_df.to_csv(metadata_root / f"{split}.csv", index=False)

    print("\nSaved metadata files:")
    print(metadata_root / "train.csv")
    print(metadata_root / "val.csv")
    print(metadata_root / "test.csv")
    print(metadata_root / "all_metadata.csv")


def print_summary(metadata):
    """
    Print summary for checking the dataset balance.
    """
    if metadata.empty:
        print("\nNo samples were prepared.")
        return

    print("\nDataset summary:")
    print(metadata.groupby(["split", "language"]).size())

    print("\nTotal per split:")
    print(metadata["split"].value_counts())

    print("\nTotal per language:")
    print(metadata["language"].value_counts())

    print("\nTotal samples:", len(metadata))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--output_root", default="data/processed")
    parser.add_argument("--metadata_root", default="data/metadata")
    parser.add_argument("--languages", nargs="+", default=None)

    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_val", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)

    args = parser.parse_args()

    if args.languages is not None:
        selected_languages = args.languages
    elif args.test_mode:
        selected_languages = ["sv"]
    else:
        selected_languages = LANGUAGES

    if args.test_mode:
        n_train = 5
        n_val = 2
        n_test = 2
    else:
        n_train = 500
        n_val = 100
        n_test = 100

    if args.n_train is not None:
        n_train = args.n_train

    if args.n_val is not None:
        n_val = args.n_val

    if args.n_test is not None:
        n_test = args.n_test

    all_rows = []

    for lang_code in selected_languages:
        rows = prepare_language(
            lang_code=lang_code,
            raw_root=args.raw_root,
            output_root=args.output_root,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
        all_rows.extend(rows)

    metadata = pd.DataFrame(all_rows)

    save_metadata(metadata, args.metadata_root)
    print_summary(metadata)


if __name__ == "__main__":
    main()