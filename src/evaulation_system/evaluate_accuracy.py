"""Reusable accuracy and per-language evaluation helpers.

This module normalizes prediction CSV files into a common structure and
computes the core metrics shared by both spoken language identification
pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score


REQUIRED_COLUMNS = {"true_label", "predicted_label"}
TIME_COLUMN_CANDIDATES = (
    "classifier_inference_time_seconds",
    "classifier_inference_time_sec",
    "classifier_inference_time",
    "end_to_end_inference_time_seconds",
    "feature_extraction_time_seconds",
    "inference_time",
    "inference_time_sec",
    "inference_time_seconds",
    "inference_time_ms",
    "inference_seconds",
    "inference_ms",
    "prediction_time",
    "prediction_time_sec",
    "prediction_time_seconds",
    "prediction_time_ms",
    "prediction_seconds",
    "prediction_ms",
    "elapsed_time",
    "elapsed_time_sec",
    "elapsed_time_seconds",
    "elapsed_time_ms",
    "latency",
    "latency_sec",
    "latency_seconds",
    "latency_ms",
    "avg_inference_time",
    "average_inference_time",
    "avg_latency",
)


@dataclass(frozen=True)
class AccuracyMetrics:
    """Container for the core evaluation results of one prediction file."""

    model_name: str
    source_path: Optional[Path]
    total_rows: int
    evaluated_rows: int
    overall_accuracy: float
    per_language_accuracy: pd.DataFrame
    best_language: str
    worst_language: str
    average_inference_time_seconds: Optional[float]
    inference_time_column: Optional[str]


def load_predictions(csv_path: str | Path) -> pd.DataFrame:
    """Load a prediction CSV and normalize its column names.

    The evaluation code expects at least ``true_label`` and ``predicted_label``.
    ``clip_id`` and timing columns are optional.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}"
        )

    return frame


def _normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean prediction rows before computing metrics."""

    cleaned = frame.copy()

    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(
            cleaned[column]
        ):
            cleaned[column] = cleaned[column].astype("string").str.strip()
            cleaned[column] = cleaned[column].replace("", pd.NA)

    evaluated = cleaned.dropna(subset=["true_label", "predicted_label"]).copy()
    evaluated["true_label"] = evaluated["true_label"].astype(str)
    evaluated["predicted_label"] = evaluated["predicted_label"].astype(str)

    return evaluated


def _detect_time_column(frame: pd.DataFrame) -> Optional[str]:
    """Find an optional timing column if the CSV provides one."""

    columns = {column.lower(): column for column in frame.columns}
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in columns:
            return columns[candidate]

    for column in frame.columns:
        normalized = column.lower()
        if "time" in normalized or "latency" in normalized or "elapsed" in normalized:
            return column

    return None


def _standardize_time_to_seconds(values: pd.Series, column_name: str) -> pd.Series:
    """Convert a timing column to seconds when the source appears to be milliseconds."""

    numeric_values = pd.to_numeric(values, errors="coerce")
    lower_name = column_name.lower()
    if "ms" in lower_name or "millisecond" in lower_name:
        return numeric_values / 1000.0

    return numeric_values


def _per_language_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate accuracy per true language label."""

    rows: list[dict[str, object]] = []
    for language, group in frame.groupby("true_label", sort=True):
        total_samples = len(group)
        correct_predictions = int((group["true_label"] == group["predicted_label"]).sum())
        language_accuracy = accuracy_score(group["true_label"], group["predicted_label"])

        rows.append(
            {
                "language": language,
                "accuracy": language_accuracy,
                "correct_predictions": correct_predictions,
                "total_samples": total_samples,
                "errors": total_samples - correct_predictions,
            }
        )

    accuracy_frame = pd.DataFrame(rows)
    if accuracy_frame.empty:
        return accuracy_frame

    return accuracy_frame.sort_values(
        by=["accuracy", "language"], ascending=[False, True]
    ).reset_index(drop=True)


def calculate_accuracy_metrics(
    frame: pd.DataFrame,
    model_name: str = "",
    source_path: str | Path | None = None,
) -> AccuracyMetrics:
    """Calculate overall and per-language accuracy for a prediction frame."""

    normalized = _normalize_prediction_frame(frame)
    if normalized.empty:
        raise ValueError("No valid prediction rows were found after cleaning the data.")

    overall_accuracy = accuracy_score(
        normalized["true_label"], normalized["predicted_label"]
    )
    per_language = _per_language_accuracy(normalized)

    if per_language.empty:
        best_language = "N/A"
        worst_language = "N/A"
    else:
        best_language = str(per_language.iloc[0]["language"])
        worst_language = str(per_language.iloc[-1]["language"])

    time_column = _detect_time_column(normalized)
    average_time_seconds: Optional[float] = None
    if time_column is not None:
        standardized = _standardize_time_to_seconds(normalized[time_column], time_column)
        standardized = standardized.dropna()
        if not standardized.empty:
            average_time_seconds = float(standardized.mean())

    source = Path(source_path) if source_path is not None else None
    return AccuracyMetrics(
        model_name=model_name,
        source_path=source,
        total_rows=len(frame),
        evaluated_rows=len(normalized),
        overall_accuracy=overall_accuracy,
        per_language_accuracy=per_language,
        best_language=best_language,
        worst_language=worst_language,
        average_inference_time_seconds=average_time_seconds,
        inference_time_column=time_column,
    )


def evaluate_accuracy(
    csv_path: str | Path,
    model_name: str = "",
) -> AccuracyMetrics:
    """Load a CSV file and calculate all accuracy metrics."""

    frame = load_predictions(csv_path)
    return calculate_accuracy_metrics(frame, model_name=model_name, source_path=csv_path)


def format_percentage(value: Optional[float]) -> str:
    """Format a numeric value as a percentage string."""

    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def format_seconds(value: Optional[float]) -> str:
    """Format a numeric value as a seconds string."""

    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.4f} sec"


def print_evaluation_summary(metrics: AccuracyMetrics) -> None:
    """Print a human-readable evaluation summary to the terminal."""

    model_label = metrics.model_name or "Model"
    source_label = str(metrics.source_path) if metrics.source_path else "<in-memory frame>"

    print("=" * 72)
    print(f"{model_label} evaluation")
    print(f"Source: {source_label}")
    print(f"Rows loaded: {metrics.total_rows}")
    print(f"Rows evaluated: {metrics.evaluated_rows}")
    print(f"Overall accuracy: {format_percentage(metrics.overall_accuracy)}")
    print(f"Best language: {metrics.best_language}")
    print(f"Worst language: {metrics.worst_language}")

    if metrics.average_inference_time_seconds is None:
        print("Average inference time: N/A")
    else:
        print(
            "Average inference time: "
            f"{format_seconds(metrics.average_inference_time_seconds)}"
        )

    print("\nPer-language accuracy:")
    if metrics.per_language_accuracy.empty:
        print("  No per-language rows available.")
    else:
        display_frame = metrics.per_language_accuracy.copy()
        display_frame["accuracy"] = display_frame["accuracy"].map(format_percentage)
        print(display_frame.to_string(index=False))
    print("=" * 72)


def save_accuracy_metrics_csv(
    metrics: AccuracyMetrics,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save overall and per-language accuracy metrics to CSV files.
    
    Returns tuple of (overall_metrics_path, per_language_accuracy_path).
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize model name for filename
    sanitized_name = metrics.model_name.lower()
    sanitized_name = sanitized_name.replace(' ', '_')
    sanitized_name = sanitized_name.replace('+', 'plus')
    sanitized_name = ''.join(c for c in sanitized_name if c.isalnum() or c == '_')
    
    # Save overall metrics as single-row CSV
    overall_data = {
        "model_name": metrics.model_name,
        "total_rows": metrics.total_rows,
        "evaluated_rows": metrics.evaluated_rows,
        "overall_accuracy": metrics.overall_accuracy,
        "overall_accuracy_percent": format_percentage(metrics.overall_accuracy),
        "best_language": metrics.best_language,
        "worst_language": metrics.worst_language,
        "average_inference_time_seconds": metrics.average_inference_time_seconds or "N/A",
    }
    overall_frame = pd.DataFrame([overall_data])
    overall_path = output_dir / f"{sanitized_name}_metrics.csv"
    overall_frame.to_csv(overall_path, index=False)
    
    # Save per-language accuracy
    per_language_path = output_dir / f"{sanitized_name}_per_language.csv"
    if not metrics.per_language_accuracy.empty:
        display_frame = metrics.per_language_accuracy.copy()
        display_frame["accuracy_percent"] = display_frame["accuracy"].map(format_percentage)
        display_frame.to_csv(per_language_path, index=False)
    else:
        empty_frame = pd.DataFrame()
        empty_frame.to_csv(per_language_path, index=False)
    
    return overall_path, per_language_path
