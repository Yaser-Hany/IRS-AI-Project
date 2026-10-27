"""Prediction module for loading a trained model and generating predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import PREDICTIONS_OUTPUT_PATH
from utils import load_pickle, setup_logger

logger = setup_logger("predict")


def predict_from_csv(model_path: Path, input_csv_path: Path, output_path: Path = PREDICTIONS_OUTPUT_PATH) -> pd.DataFrame:
    """Load trained model and predict RSSI for input feature CSV.

    Args:
        model_path: Path to serialized pipeline model.
        input_csv_path: CSV path containing feature columns.
        output_path: Path to save prediction output.

    Returns:
        DataFrame including predictions.
    """
    logger.info("Loading model from: %s", model_path)
    model = load_pickle(model_path)

    logger.info("Loading input data from: %s", input_csv_path)
    input_df = pd.read_csv(input_csv_path)

    preds = model.predict(input_df)
    result = input_df.copy()
    result["predicted_rssi_dbm"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Predictions saved to: %s", output_path)

    return result
