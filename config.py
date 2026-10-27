"""Configuration settings for RF Signal Strength Prediction project."""

from pathlib import Path

# Reproducibility
RANDOM_STATE: int = 42

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
SCREENSHOTS_DIR: Path = PROJECT_ROOT / "screenshots"

DATASET_PATH: Path = DATA_DIR / "rf_dataset.csv"
SAMPLE_PREDICTION_PATH: Path = DATA_DIR / "sample_prediction.csv"
PREDICTIONS_OUTPUT_PATH: Path = OUTPUTS_DIR / "prediction.csv"
METRICS_OUTPUT_PATH: Path = OUTPUTS_DIR / "metrics.txt"

# Dataset generation
N_SAMPLES: int = 5000

# Train/test split
TEST_SIZE: float = 0.2

# Features and target
NUMERICAL_FEATURES = [
    "distance_m",
    "frequency_mhz",
    "transmit_power_dbm",
    "antenna_gain_dbi",
    "obstacle_loss_db",
]
CATEGORICAL_FEATURES = ["environment_type"]
TARGET: str = "rssi_dbm"

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Model hyperparameters
MODEL_CONFIG = {
    "linear_regression": {},
    "decision_tree": {
        "max_depth": 12,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 18,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.9,
        "random_state": RANDOM_STATE,
    },
}

# Environment options used in synthetic generation
ENVIRONMENT_TYPES = ["Urban", "Suburban", "Rural", "Indoor", "Industrial"]

# Matplotlib style
PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
FIGURE_DPI: int = 300
