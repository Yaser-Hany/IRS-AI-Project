"""Main entry point for RF Signal Strength Prediction project.

Run:
    python main.py
"""

from __future__ import annotations

from config import (
    DATASET_PATH,
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    SAMPLE_PREDICTION_PATH,
    SCREENSHOTS_DIR,
)
from dataset import create_sample_prediction_input, load_or_create_dataset
from predict import predict_from_csv
from train import train_and_evaluate
from utils import ensure_directories, setup_logger


logger = setup_logger("main")


def main() -> None:
    """Execute end-to-end dataset generation, training, evaluation, and prediction."""
    ensure_directories(DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SCREENSHOTS_DIR)

    logger.info("Loading or generating RF dataset.")
    df = load_or_create_dataset(DATASET_PATH)

    create_sample_prediction_input(df, SAMPLE_PREDICTION_PATH, n_rows=25)

    logger.info("Training and evaluating models.")
    _, metrics, best_model_name = train_and_evaluate(df)

    logger.info("Best model selected: %s", best_model_name)
    best_model_path = MODELS_DIR / f"{best_model_name}.pkl"

    logger.info("Generating predictions from sample input.")
    _ = predict_from_csv(best_model_path, SAMPLE_PREDICTION_PATH)

    logger.info("Pipeline completed successfully.")
    for model_name, model_metrics in metrics.items():
        logger.info("%s => RMSE: %.4f | R2: %.4f", model_name, model_metrics["RMSE"], model_metrics["R2"])


if __name__ == "__main__":
    main()
