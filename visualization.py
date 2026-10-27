"""Visualization module for RF signal strength prediction project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURE_DPI, PLOT_STYLE


plt.style.use(PLOT_STYLE)


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_rssi_vs_distance(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["distance_m"], df["rssi_dbm"], alpha=0.4, s=12)
    plt.xlabel("Distance (m)")
    plt.ylabel("RSSI (dBm)")
    plt.title("RSSI vs Distance")
    _save_fig(out_path)


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=12)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5)
    plt.xlabel("Actual RSSI (dBm)")
    plt.ylabel("Predicted RSSI (dBm)")
    plt.title(f"Actual vs Predicted - {model_name}")
    _save_fig(out_path)


def plot_residuals(y_true, y_pred, model_name: str, out_path: Path) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.5, s=12)
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted RSSI (dBm)")
    plt.ylabel("Residuals (dBm)")
    plt.title(f"Residual Plot - {model_name}")
    _save_fig(out_path)


def plot_feature_importance(feature_importance: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    feature_importance.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    _save_fig(out_path)


def plot_correlation_matrix(df: pd.DataFrame, out_path: Path) -> None:
    corr = df.select_dtypes(include=["number"]).corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    _save_fig(out_path)


def plot_model_comparison(metrics_by_model: Dict[str, Dict[str, float]], out_path: Path) -> None:
    models = list(metrics_by_model.keys())
    rmse_values = [metrics_by_model[m]["RMSE"] for m in models]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, rmse_values)
    plt.ylabel("RMSE")
    plt.title("Model Comparison (Lower RMSE is Better)")
    plt.xticks(rotation=20)

    for bar, val in zip(bars, rmse_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save_fig(out_path)


def plot_signal_strength_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    # Bin distance and frequency for a 2D RSSI map
    work = df.copy()
    work["distance_bin"] = pd.cut(work["distance_m"], bins=20)
    work["frequency_bin"] = pd.cut(work["frequency_mhz"], bins=10)

    pivot = work.pivot_table(
        index="distance_bin",
        columns="frequency_bin",
        values="rssi_dbm",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 6))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Mean RSSI (dBm)")
    plt.xlabel("Frequency bins")
    plt.ylabel("Distance bins")
    plt.title("Signal Strength Heatmap")
    _save_fig(out_path)
