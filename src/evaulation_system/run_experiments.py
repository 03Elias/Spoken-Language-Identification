"""Run the expanded DT2119 SLI experiment grid from cached features.

This script assumes MFCC and wav2vec features have already been extracted.
It trains classifiers, evaluates on validation/test splits.
"""

from __future__ import annotations 

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .confusion_matrix import save_confusion_matrix_plot
    from .evaluate_accuracy import calculate_accuracy_metrics
except ImportError:  # pragma: no cover
    from confusion_matrix import save_confusion_matrix_plot
    from evaluate_accuracy import calculate_accuracy_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MFCC_DIR = PROJECT_ROOT / "src" / "mfcc_system"
WAV2VEC_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "src" / "evaluation_outputs"

LANGUAGE_ORDER = ["ar", "es", "no", "pt", "sv"]
CLASSIFIERS = ["logreg", "svm_rbf"]
WAV2VEC_LAYERS = [1, 3, 6, 9, 12]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    feature_family: str
    feature_type: str
    classifier: str
    feature_dim: int
    layer: int | None = None


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

    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def mfcc_feature_path(split: str, feature_type: str) -> Path:
    candidate = MFCC_DIR / f"X_{split}_{feature_type}.npy"
    if candidate.exists():
        return candidate
    if feature_type == "baseline":
        return MFCC_DIR / f"X_{split}.npy"
    return candidate


def mfcc_label_path(split: str, feature_type: str) -> Path:
    candidate = MFCC_DIR / f"y_{split}_{feature_type}.npy"
    if candidate.exists():
        return candidate
    if feature_type == "baseline":
        return MFCC_DIR / f"y_{split}.npy"
    return candidate


def wav2vec_feature_path(split: str, layer: int) -> Path:
    candidate = WAV2VEC_DIR / f"embeddings_layer_{layer}_{split}.npy"
    if candidate.exists():
        return candidate
    if layer == 12:
        return WAV2VEC_DIR / f"embeddings_{split}.npy"
    return candidate


def load_feature_pack(spec: ExperimentSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if spec.feature_family == "mfcc":
        assert spec.layer is None
        X_train = np.load(mfcc_feature_path("train", spec.feature_type))
        y_train = np.load(mfcc_label_path("train", spec.feature_type))
        X_val = np.load(mfcc_feature_path("val", spec.feature_type))
        y_val = np.load(mfcc_label_path("val", spec.feature_type))
        X_test = np.load(mfcc_feature_path("test", spec.feature_type))
        y_test = np.load(mfcc_label_path("test", spec.feature_type))
        clip_path = MFCC_DIR / f"clip_ids_test_{spec.feature_type}.npy"
        clip_ids = np.load(clip_path) if clip_path.exists() else np.arange(len(y_test)).astype(str)
        return X_train, y_train, X_val, y_val, X_test, y_test, clip_ids

    assert spec.layer is not None
    X_train = np.load(wav2vec_feature_path("train", spec.layer))
    y_train = np.load(WAV2VEC_DIR / "labels_train.npy")
    X_val = np.load(wav2vec_feature_path("val", spec.layer))
    y_val = np.load(WAV2VEC_DIR / "labels_val.npy")
    X_test = np.load(wav2vec_feature_path("test", spec.layer))
    y_test = np.load(WAV2VEC_DIR / "labels_test.npy")
    clip_ids = np.load(WAV2VEC_DIR / "clip_ids_test.npy")
    return X_train, y_train, X_val, y_val, X_test, y_test, clip_ids


def mean_feature_time_seconds(spec: ExperimentSpec) -> float | None:
    if spec.feature_family == "mfcc":
        path = MFCC_DIR / f"feature_times_test_{spec.feature_type}.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
    else:
        path = WAV2VEC_DIR / "feature_times_wav2vec_test.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        frame = frame[frame["feature_layer"].astype(str) == str(spec.layer)]

    if frame.empty or "feature_extraction_time_seconds" not in frame.columns:
        return None
    return float(frame["feature_extraction_time_seconds"].mean())


def specs() -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []
    for feature_type, feature_dim in [("baseline", 26), ("temporal", 78)]:
        for classifier in CLASSIFIERS:
            experiments.append(
                ExperimentSpec(
                    experiment_id=f"mfcc_{feature_type}_{classifier}",
                    feature_family="mfcc",
                    feature_type=feature_type,
                    classifier=classifier,
                    feature_dim=feature_dim,
                )
            )

    for layer in WAV2VEC_LAYERS:
        for classifier in CLASSIFIERS:
            experiments.append(
                ExperimentSpec(
                    experiment_id=f"wav2vec_layer_{layer}_{classifier}",
                    feature_family="wav2vec",
                    feature_type="wav2vec",
                    classifier=classifier,
                    feature_dim=768,
                    layer=layer,
                )
            )
    return experiments


def required_paths(spec: ExperimentSpec) -> list[Path]:
    if spec.feature_family == "mfcc":
        return [
            mfcc_feature_path(split, spec.feature_type)
            for split in ["train", "val", "test"]
        ] + [
            mfcc_label_path(split, spec.feature_type)
            for split in ["train", "val", "test"]
        ]
    assert spec.layer is not None
    return [
        wav2vec_feature_path(split, spec.layer)
        for split in ["train", "val", "test"]
    ] + [
        WAV2VEC_DIR / f"labels_{split}.npy"
        for split in ["train", "val", "test"]
    ]


def run_one(spec: ExperimentSpec) -> tuple[dict[str, object], pd.DataFrame]:
    X_train, y_train, X_val, y_val, X_test, y_test, clip_ids = load_feature_pack(spec)
    model = build_classifier(spec.classifier)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)

    start = time.perf_counter()
    test_preds = model.predict(X_test)
    classifier_elapsed = time.perf_counter() - start
    classifier_per_clip = classifier_elapsed / len(X_test) if len(X_test) else 0.0
    feature_per_clip = mean_feature_time_seconds(spec)
    end_to_end = (
        feature_per_clip + classifier_per_clip if feature_per_clip is not None else None
    )

    pred_frame = pd.DataFrame(
        {
            "experiment_id": spec.experiment_id,
            "feature_type": (
                f"mfcc_{spec.feature_type}"
                if spec.feature_family == "mfcc"
                else "wav2vec"
            ),
            "feature_layer": "" if spec.layer is None else spec.layer,
            "classifier": spec.classifier,
            "clip_id": clip_ids,
            "true_label": y_test,
            "predicted_label": test_preds,
            "feature_extraction_time_seconds": feature_per_clip,
            "classifier_inference_time_seconds": classifier_per_clip,
            "end_to_end_inference_time_seconds": end_to_end,
        }
    )
    metrics = calculate_accuracy_metrics(pred_frame, model_name=spec.experiment_id)

    prediction_path = OUTPUT_DIR / f"predictions_{spec.experiment_id}.csv"
    pred_frame.to_csv(prediction_path, index=False)

    row = {
        "experiment_id": spec.experiment_id,
        "feature_family": spec.feature_family,
        "feature_type": (
            f"MFCC {spec.feature_type}"
            if spec.feature_family == "mfcc"
            else "wav2vec"
        ),
        "wav2vec_layer": spec.layer if spec.layer is not None else "",
        "classifier": spec.classifier,
        "feature_dimension": spec.feature_dim,
        "validation_accuracy": val_accuracy,
        "test_accuracy": metrics.overall_accuracy,
        "best_language": metrics.best_language,
        "worst_language": metrics.worst_language,
        "feature_extraction_time_seconds": feature_per_clip,
        "classifier_inference_time_seconds": classifier_per_clip,
        "end_to_end_inference_time_seconds": end_to_end,
        "prediction_csv": str(prediction_path),
    }
    return row, pred_frame


def best_worst_note(row: pd.Series) -> str:
    return f"Best: {row['best_language']}; worst: {row['worst_language']}"


def best_wav2vec_layer(summary: pd.DataFrame) -> int | None:
    wav2vec_rows = summary[summary["feature_family"] == "wav2vec"].copy()
    if wav2vec_rows.empty:
        return None
    best_row = wav2vec_rows.sort_values("test_accuracy", ascending=False).iloc[0]
    return int(best_row["wav2vec_layer"])


def main_comparison_ids(summary: pd.DataFrame) -> set[str]:
    ids = {
        "mfcc_baseline_logreg",
        "mfcc_baseline_svm_rbf",
        "mfcc_temporal_logreg",
        "mfcc_temporal_svm_rbf",
        "wav2vec_layer_12_logreg",
        "wav2vec_layer_12_svm_rbf",
    }
    best_layer = best_wav2vec_layer(summary)
    if best_layer is not None:
        ids.update(
            {
                f"wav2vec_layer_{best_layer}_logreg",
                f"wav2vec_layer_{best_layer}_svm_rbf",
            }
        )
    return ids


def write_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    main_ids = main_comparison_ids(summary)
    best_layer = best_wav2vec_layer(summary)
    table = summary[summary["experiment_id"].isin(main_ids)].copy()
    table["Feature type"] = table.apply(
        lambda row: (
            "wav2vec final layer"
            if row["feature_family"] == "wav2vec" and int(row["wav2vec_layer"]) == 12
            else (
                f"wav2vec best layer ({best_layer})"
                if row["feature_family"] == "wav2vec"
                else row["feature_type"]
            )
        ),
        axis=1,
    )
    table["Classifier"] = table["classifier"].map(
        {"logreg": "Logistic Regression", "svm_rbf": "RBF-SVM"}
    )
    table["Feature dimension"] = table["feature_dimension"]
    table["Accuracy"] = table["test_accuracy"]
    table["Best/worst language"] = table.apply(best_worst_note, axis=1)
    table["Notes"] = table.apply(
        lambda row: "Temporal dynamics included"
        if row["experiment_id"].startswith("mfcc_temporal")
        else (
            "Final hidden layer"
            if row["feature_family"] == "wav2vec" and int(row["wav2vec_layer"]) == 12
            else (
                "Best layer from layer-wise analysis"
                if row["feature_family"] == "wav2vec"
                else "Mean/std pooling"
            )
        ),
        axis=1,
    )
    output = table[
        [
            "Feature type",
            "Classifier",
            "Feature dimension",
            "Accuracy",
            "Best/worst language",
            "Notes",
        ]
    ]
    output.to_csv(OUTPUT_DIR / "main_cross_classifier_comparison.csv", index=False)
    output.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    return output


def write_layer_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[summary["feature_family"] == "wav2vec"].copy()
    table["layer_sort"] = table["wav2vec_layer"].astype(int)
    table["wav2vec layer"] = table["wav2vec_layer"].map(
        lambda layer: "12/final" if int(layer) == 12 else int(layer)
    )
    table["Classifier"] = table["classifier"].map(
        {"logreg": "Logistic Regression", "svm_rbf": "RBF-SVM"}
    )
    table["Embedding dimension"] = table["feature_dimension"]
    table["Accuracy"] = table["test_accuracy"]
    table["Best language"] = table["best_language"]
    table["Worst language"] = table["worst_language"]
    output = table[
        [
            "layer_sort",
            "wav2vec layer",
            "Classifier",
            "Embedding dimension",
            "Accuracy",
            "Best language",
            "Worst language",
        ]
    ].sort_values(["layer_sort", "Classifier"])
    output = output.drop(columns=["layer_sort"])
    output.to_csv(OUTPUT_DIR / "wav2vec_layerwise_analysis.csv", index=False)
    return output


def write_per_language_table(summary: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def best_id(mask: pd.Series) -> str | None:
        candidates = summary[mask].sort_values("test_accuracy", ascending=False)
        if candidates.empty:
            return None
        return str(candidates.iloc[0]["experiment_id"])

    baseline_id = best_id(summary["experiment_id"].str.startswith("mfcc_baseline"))
    temporal_id = best_id(summary["experiment_id"].str.startswith("mfcc_temporal"))
    wav_best_id = best_id(summary["feature_family"] == "wav2vec")
    final_id = best_id(summary["experiment_id"].str.startswith("wav2vec_layer_12"))

    columns = {
        "MFCC baseline": baseline_id,
        "MFCC temporal": temporal_id,
        "wav2vec best layer": wav_best_id,
        "wav2vec final layer": final_id,
    }
    rows: list[dict[str, object]] = []
    for language in LANGUAGE_ORDER:
        row: dict[str, object] = {"language": language}
        for label, experiment_id in columns.items():
            if experiment_id is None:
                row[label] = np.nan
                continue
            frame = predictions[experiment_id]
            lang_frame = frame[frame["true_label"].astype(str) == language]
            if lang_frame.empty:
                row[label] = np.nan
            else:
                row[label] = accuracy_score(
                    lang_frame["true_label"], lang_frame["predicted_label"]
                )
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "per_language_comparison.csv", index=False)
    return table


def plot_cross_classifier(summary: pd.DataFrame) -> None:
    main = summary[summary["experiment_id"].isin(main_comparison_ids(summary))].copy()
    main["label"] = main["experiment_id"].str.replace("_", "\n", regex=False)
    plt.figure(figsize=(11, 5))
    plt.bar(main["label"], main["test_accuracy"])
    plt.ylabel("Accuracy")
    plt.xlabel("System configuration")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cross_classifier_accuracy.png", dpi=200)
    plt.close()


def plot_layer_accuracy(summary: pd.DataFrame) -> None:
    wav = summary[summary["feature_family"] == "wav2vec"].copy()
    plt.figure(figsize=(8, 5))
    for classifier, group in wav.groupby("classifier"):
        group = group.sort_values("wav2vec_layer")
        plt.plot(group["wav2vec_layer"], group["test_accuracy"], marker="o", label=classifier)
    plt.xlabel("wav2vec layer")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(WAV2VEC_LAYERS)
    plt.legend(title="Classifier")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wav2vec_layer_accuracy.png", dpi=200)
    plt.close()


def write_best_confusions(summary: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> None:
    mfcc_id = str(
        summary[summary["feature_family"] == "mfcc"]
        .sort_values("test_accuracy", ascending=False)
        .iloc[0]["experiment_id"]
    )
    temporal_mfcc = summary[summary["experiment_id"].str.startswith("mfcc_temporal")]
    temporal_mfcc_id = (
        str(temporal_mfcc.sort_values("test_accuracy", ascending=False).iloc[0]["experiment_id"])
        if not temporal_mfcc.empty
        else None
    )
    wav_id = str(
        summary[summary["feature_family"] == "wav2vec"]
        .sort_values("test_accuracy", ascending=False)
        .iloc[0]["experiment_id"]
    )
    save_confusion_matrix_plot(
        predictions[mfcc_id],
        OUTPUT_DIR / "best_mfcc_confusion_matrix.png",
        title=f"Best MFCC System: {mfcc_id}",
    )
    if temporal_mfcc_id is not None:
        save_confusion_matrix_plot(
            predictions[temporal_mfcc_id],
            OUTPUT_DIR / "temporal_mfcc_confusion_matrix.png",
            title=f"Temporal MFCC System: {temporal_mfcc_id}",
        )
    save_confusion_matrix_plot(
        predictions[wav_id],
        OUTPUT_DIR / "best_wav2vec_confusion_matrix.png",
        title=f"Best wav2vec System: {wav_id}",
    )


def run_all(allow_missing: bool = False) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}

    for spec in specs():
        missing = [path for path in required_paths(spec) if not path.exists()]
        if missing:
            message = (
                f"Missing cached inputs for {spec.experiment_id}: "
                + ", ".join(str(path) for path in missing)
            )
            if allow_missing:
                print(f"Skipping: {message}")
                continue
            raise FileNotFoundError(message)

        print(f"Running {spec.experiment_id}...")
        row, pred_frame = run_one(spec)
        rows.append(row)
        predictions[spec.experiment_id] = pred_frame

    if not rows:
        raise RuntimeError("No experiments were run.")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "all_experiment_metrics.csv", index=False)

    write_main_table(summary)
    write_layer_table(summary)
    write_per_language_table(summary, predictions)
    plot_cross_classifier(summary)
    plot_layer_accuracy(summary)
    write_best_confusions(summary, predictions)

    print(f"Saved evaluation artifacts to {OUTPUT_DIR}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all cached SLI experiments.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip experiments whose feature caches are not present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all(allow_missing=args.allow_missing)


if __name__ == "__main__":
    main()
