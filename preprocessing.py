"""Data preprocessing with sklearn pipelines."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into feature matrix and target vector.

    Args:
        df: Input dataframe.

    Returns:
        X dataframe and y series.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()
    return X, y


def train_test_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform train/test split.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Test ratio.
        random_state: Seed.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing transformer for numerical and categorical features.

    Returns:
        ColumnTransformer object.
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
