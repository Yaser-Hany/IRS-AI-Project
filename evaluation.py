"""Evaluation metrics and reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with MAE, MSE, RMSE, and R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def save_metrics_report(metrics_by_model: Dict[str, Dict[str, float]], path: Path) -> None:
    """Save model metrics summary to text file.

    Args:
        metrics_by_model: Nested dict containing metrics per model.
        path: Output text path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["RF Signal Strength Prediction - Metrics Report", "=" * 50, ""]

    for model_name, metrics in metrics_by_model.items():
        lines.append(f"Model: {model_name}")
        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.6f}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
