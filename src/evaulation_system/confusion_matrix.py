"""Reusable confusion matrix plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback for minimal environments
    sns = None


def save_confusion_matrix_plot(
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    normalize: bool = False,
) -> Path:
    """Create and save a confusion matrix heatmap as a PNG file."""

    required_columns = {"true_label", "predicted_label"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Data frame is missing required columns: {sorted(missing_columns)}"
        )

    cleaned = frame.dropna(subset=["true_label", "predicted_label"]).copy()
    if cleaned.empty:
        raise ValueError("Cannot plot a confusion matrix from an empty data frame.")

    cleaned["true_label"] = cleaned["true_label"].astype(str)
    cleaned["predicted_label"] = cleaned["predicted_label"].astype(str)

    labels = sorted(
        set(cleaned["true_label"].unique()).union(set(cleaned["predicted_label"].unique()))
    )
    matrix = confusion_matrix(
        cleaned["true_label"],
        cleaned["predicted_label"],
        labels=labels,
        normalize="true" if normalize else None,
    )

    figure_size = max(7.0, 0.7 * len(labels) + 4.0)
    plt.figure(figsize=(figure_size, figure_size))
    fmt = ".2f" if normalize else "d"

    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=normalize,
            square=True,
            linewidths=0.5,
            linecolor="white",
        )
    else:
        plt.imshow(matrix, cmap="Blues")
        plt.colorbar() if normalize else None
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        threshold = matrix.max() / 2.0 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                annotation = f"{value:.2f}" if normalize else f"{int(value)}"
                text_color = "white" if value > threshold else "black"
                plt.text(
                    column_index,
                    row_index,
                    annotation,
                    ha="center",
                    va="center",
                    color=text_color,
                )

    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    return output_file
