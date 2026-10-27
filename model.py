"""Model definitions and training utilities."""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from config import MODEL_CONFIG


def build_models() -> Dict[str, object]:
    """Instantiate supported regressors.

    Returns:
        Dictionary of model name to estimator.
    """
    return {
        "linear_regression": LinearRegression(**MODEL_CONFIG["linear_regression"]),
        "decision_tree": DecisionTreeRegressor(**MODEL_CONFIG["decision_tree"]),
        "random_forest": RandomForestRegressor(**MODEL_CONFIG["random_forest"]),
        "gradient_boosting": GradientBoostingRegressor(
            **MODEL_CONFIG["gradient_boosting"]
        ),
    }


def build_pipeline(preprocessor, estimator) -> Pipeline:
    """Build model pipeline.

    Args:
        preprocessor: Feature preprocessing transformer.
        estimator: Regression model.

    Returns:
        Full sklearn pipeline.
    """
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
