"""Compare MFCC + SVM against wav2vec 2.0 + Linear Classifier."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

try:  # Allow running both as a module and as a direct script.
    from .confusion_matrix import save_confusion_matrix_plot
    from .evaluate_accuracy import (
        AccuracyMetrics,
        calculate_accuracy_metrics,
        format_percentage,
        format_seconds,
        load_predictions,
        print_evaluation_summary,
        save_accuracy_metrics_csv,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from confusion_matrix import save_confusion_matrix_plot
    from evaluate_accuracy import (
        AccuracyMetrics,
        calculate_accuracy_metrics,
        format_percentage,
        format_seconds,
        load_predictions,
        print_evaluation_summary,
        save_accuracy_metrics_csv,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and compare MFCC + SVM against wav2vec 2.0 + Linear Classifier."
        )
    )
    parser.add_argument(
        "--mfcc",
        required=True,
        help="Path to the MFCC + SVM prediction CSV file.",
    )
    parser.add_argument(
        "--wav2vec",
        required=True,
        help="Path to the wav2vec 2.0 + Linear Classifier prediction CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_outputs",
        help="Directory where confusion matrix PNG files will be saved.",
    )
    parser.add_argument(
        "--normalize-confusion-matrix",
        action="store_true",
        help="Save normalized confusion matrices instead of raw counts.",
    )
    return parser.parse_args(argv)


def _warn_if_row_counts_differ(mfcc_rows: int, wav2vec_rows: int) -> None:
    """Warn when the two evaluation files contain different row counts."""

    if mfcc_rows != wav2vec_rows:
        message = (
            "The two test sets are not the same size: "
            f"MFCC has {mfcc_rows} rows, wav2vec has {wav2vec_rows} rows."
        )
        warnings.warn(message, stacklevel=2)
        print(f"WARNING: {message}")


def _report_clip_id_status(frame: pd.DataFrame, model_name: str) -> None:
    """Tell the user whether the optional clip_id column was present."""

    if "clip_id" not in frame.columns:
        print(f"NOTE: {model_name} predictions do not include clip_id; continuing without it.")


def _evaluate_model(csv_path: str | Path, model_name: str) -> tuple[AccuracyMetrics, pd.DataFrame]:
    """Load a CSV, evaluate it, and return both the metrics and the cleaned frame."""

    frame = load_predictions(csv_path)
    metrics = calculate_accuracy_metrics(frame, model_name=model_name, source_path=csv_path)
    return metrics, frame


def _build_comparison_table(
    mfcc_metrics: AccuracyMetrics,
    wav2vec_metrics: AccuracyMetrics,
) -> pd.DataFrame:
    """Create a final side-by-side comparison table."""

    rows = [
        {
            "Metric": "Accuracy",
            "MFCC + SVM": format_percentage(mfcc_metrics.overall_accuracy),
            "wav2vec + Linear": format_percentage(wav2vec_metrics.overall_accuracy),
        },
        {
            "Metric": "Best language",
            "MFCC + SVM": mfcc_metrics.best_language,
            "wav2vec + Linear": wav2vec_metrics.best_language,
        },
        {
            "Metric": "Worst language",
            "MFCC + SVM": mfcc_metrics.worst_language,
            "wav2vec + Linear": wav2vec_metrics.worst_language,
        },
        {
            "Metric": "Avg inference time",
            "MFCC + SVM": format_seconds(mfcc_metrics.average_inference_time_seconds),
            "wav2vec + Linear": format_seconds(wav2vec_metrics.average_inference_time_seconds),
        },
    ]

    return pd.DataFrame(rows)


def _print_confusion_matrix_status(output_path: Path) -> None:
    """Print the path of a generated confusion matrix plot."""

    print(f"Saved confusion matrix: {output_path}")


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and comparison workflow."""

    args = parse_args(argv)

    mfcc_path = Path(args.mfcc)
    wav2vec_path = Path(args.wav2vec)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mfcc_metrics, mfcc_frame = _evaluate_model(mfcc_path, "MFCC + SVM")
    wav2vec_metrics, wav2vec_frame = _evaluate_model(wav2vec_path, "wav2vec + Linear")

    _warn_if_row_counts_differ(mfcc_metrics.total_rows, wav2vec_metrics.total_rows)
    _report_clip_id_status(mfcc_frame, "MFCC + SVM")
    _report_clip_id_status(wav2vec_frame, "wav2vec + Linear")

    print_evaluation_summary(mfcc_metrics)
    print_evaluation_summary(wav2vec_metrics)

    # Save individual model metrics to CSV
    mfcc_overall_csv, mfcc_per_lang_csv = save_accuracy_metrics_csv(
        mfcc_metrics, output_dir
    )
    wav2vec_overall_csv, wav2vec_per_lang_csv = save_accuracy_metrics_csv(
        wav2vec_metrics, output_dir
    )
    print(f"\nSaved MFCC + SVM metrics: {mfcc_overall_csv}")
    print(f"Saved MFCC + SVM per-language accuracy: {mfcc_per_lang_csv}")
    print(f"Saved wav2vec + Linear metrics: {wav2vec_overall_csv}")
    print(f"Saved wav2vec + Linear per-language accuracy: {wav2vec_per_lang_csv}")

    mfcc_plot = save_confusion_matrix_plot(
        mfcc_frame,
        output_dir / "mfcc_svm_confusion_matrix.png",
        title="MFCC + SVM Confusion Matrix",
        normalize=args.normalize_confusion_matrix,
    )
    wav2vec_plot = save_confusion_matrix_plot(
        wav2vec_frame,
        output_dir / "wav2vec_linear_confusion_matrix.png",
        title="wav2vec + Linear Confusion Matrix",
        normalize=args.normalize_confusion_matrix,
    )

    _print_confusion_matrix_status(mfcc_plot)
    _print_confusion_matrix_status(wav2vec_plot)

    comparison_table = _build_comparison_table(mfcc_metrics, wav2vec_metrics)

    print("\nFinal comparison table:")
    print(comparison_table.to_string(index=False))

    # Save comparison table as CSV
    comparison_csv_path = output_dir / "comparison_table.csv"
    comparison_table.to_csv(comparison_csv_path, index=False)
    print(f"Saved comparison table: {comparison_csv_path}")

    if mfcc_metrics.average_inference_time_seconds is None:
        print("\nMFCC + SVM did not include an inference-time column.")
    if wav2vec_metrics.average_inference_time_seconds is None:
        print("wav2vec + Linear did not include an inference-time column.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
