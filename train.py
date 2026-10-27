"""Training workflow for RF Signal Strength Prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from config import (
    METRICS_OUTPUT_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from evaluation import regression_metrics, save_metrics_report
from model import build_models, build_pipeline
from preprocessing import build_preprocessor, split_features_target, train_test_data
from utils import save_pickle, setup_logger
from visualization import (
    plot_actual_vs_predicted,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    plot_rssi_vs_distance,
    plot_signal_strength_heatmap,
)

logger = setup_logger("train")


def _extract_feature_importance(trained_pipeline, X_columns) -> pd.Series:
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    model = trained_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = abs(coef.flatten() if hasattr(coef, "flatten") else coef)
    else:
        importances = [0.0 for _ in feature_names]

    return pd.Series(importances, index=feature_names)


def train_and_evaluate(df: pd.DataFrame) -> Tuple[dict, Dict[str, Dict[str, float]], str]:
    """Train all models, evaluate, and persist artifacts.

    Args:
        df: Input RF dataframe.

    Returns:
        Tuple of trained_models, metrics_by_model, best_model_name.
    """
    logger.info("Starting training pipeline.")

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_data(X, y)

    preprocessor = build_preprocessor()
    model_registry = build_models()

    trained_models = {}
    metrics_by_model: Dict[str, Dict[str, float]] = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Static dataset visualizations
    plot_rssi_vs_distance(df, OUTPUTS_DIR / "scatter.png")
    plot_correlation_matrix(df, OUTPUTS_DIR / "correlation_matrix.png")
    plot_signal_strength_heatmap(df, OUTPUTS_DIR / "heatmap.png")

    for model_name, estimator in model_registry.items():
        logger.info("Training model: %s", model_name)
        pipeline = build_pipeline(preprocessor=preprocessor, estimator=estimator)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        metrics_by_model[model_name] = metrics
        trained_models[model_name] = pipeline

        model_path = MODELS_DIR / f"{model_name}.pkl"
        save_pickle(pipeline, model_path)

        plot_actual_vs_predicted(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            out_path=OUTPUTS_DIR / f"actual_vs_predicted_{model_name}.png",
        )
        plot_residuals(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            out_path=OUTPUTS_DIR / f"residual_plot_{model_name}.png",
        )

    best_model_name = min(metrics_by_model, key=lambda name: metrics_by_model[name]["RMSE"])
    logger.info("Best model by RMSE: %s", best_model_name)

    best_importance = _extract_feature_importance(trained_models[best_model_name], X.columns)
    plot_feature_importance(best_importance, OUTPUTS_DIR / "feature_importance.png")
    plot_model_comparison(metrics_by_model, OUTPUTS_DIR / "model_comparison.png")

    save_metrics_report(metrics_by_model, METRICS_OUTPUT_PATH)

    return trained_models, metrics_by_model, best_model_name
