"""Synthetic RF dataset generation and loading module."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DATASET_PATH,
    ENVIRONMENT_TYPES,
    N_SAMPLES,
    RANDOM_STATE,
)


def _environment_loss_map() -> dict[str, float]:
    """Return baseline propagation loss adjustments by environment."""
    return {
        "Urban": 12.0,
        "Suburban": 7.0,
        "Rural": 3.5,
        "Indoor": 18.0,
        "Industrial": 14.0,
    }


def generate_synthetic_rf_dataset(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a realistic synthetic RF dataset with 3D IRS coordinates.

    The RSSI model combines free-space path loss (FSPL) using 3D spatial distances
    with environment and obstacle penalties, plus stochastic fading noise.
    """
    rng = np.random.default_rng(random_state)

    # 1. تحديد إحداثيات 3D ثابتة لمحطة الإرسال والـ IRS
    tx_pos = np.array([0.0, 0.0, 15.0])    # محطة الإرسال على ارتفاع 15 متر
    irs_pos = np.array([50.0, 50.0, 5.0])  # الـ IRS على بعد 50 متر وارتفاع 5 متر

    # 2. توليد إحداثيات 3D عشوائية للمستخدمين (Rx)
    rx_x = rng.uniform(0, 2000, size=n_samples)
    rx_y = rng.uniform(0, 2000, size=n_samples)
    rx_z = rng.uniform(1.0, 1.8, size=n_samples) # ارتفاع الهاتف

    # 3. حساب المسافات ثلاثية الأبعاد
    dist_tx_irs = np.sqrt(np.sum((tx_pos - irs_pos)**2))
    dist_irs_rx = np.sqrt((rx_x - irs_pos[0])**2 + (rx_y - irs_pos[1])**2 + (rx_z - irs_pos[2])**2)
    
    # المسافة الكلية (من المحطة للـ IRS ثم للمستخدم)
    distance_m = dist_tx_irs + dist_irs_rx

    frequency_mhz = rng.choice([700, 800, 900, 1800, 2100, 2600, 3500], size=n_samples)
    transmit_power_dbm = rng.uniform(20, 46, size=n_samples)
    antenna_gain_dbi = rng.uniform(0, 18, size=n_samples)
    obstacle_loss_db = rng.uniform(0, 30, size=n_samples)
    environment_type = rng.choice(ENVIRONMENT_TYPES, size=n_samples)

    # FSPL in dB with distance in km and frequency in MHz
    distance_km = np.clip(distance_m / 1000.0, 0.01, None)
    fspl_db = 32.44 + 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz)

    env_loss_lookup = _environment_loss_map()
    env_loss_db = np.array([env_loss_lookup[e] for e in environment_type])

    shadowing_noise = rng.normal(0, 4.5, size=n_samples)

    rssi_dbm = (
        transmit_power_dbm
        + antenna_gain_dbi
        - fspl_db
        - obstacle_loss_db
        - env_loss_db
        + shadowing_noise
    )

    # دمج إحداثيات الـ 3D كـ Features جديدة ليتدرب عليها الموديل
    data = pd.DataFrame(
        {
            "rx_x": rx_x,
            "rx_y": rx_y,
            "rx_z": rx_z,
            "distance_m": distance_m,
            "frequency_mhz": frequency_mhz,
            "transmit_power_dbm": transmit_power_dbm,
            "antenna_gain_dbi": antenna_gain_dbi,
            "obstacle_loss_db": obstacle_loss_db,
            "environment_type": environment_type,
            "rssi_dbm": rssi_dbm,
        }
    )

    return data


def save_dataset(df: pd.DataFrame, path: Path = DATASET_PATH) -> None:
    """Save dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_or_create_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load dataset from path, creating synthetic data if unavailable."""
    if path.exists():
        return pd.read_csv(path)

    df = generate_synthetic_rf_dataset()
    save_dataset(df, path)
    return df


def create_sample_prediction_input(
    df: pd.DataFrame,
    out_path: Optional[Path] = None,
    n_rows: int = 20,
) -> pd.DataFrame:
    """Create sample input file for inference."""
    sample = df.drop(columns=["rssi_dbm"]).sample(n=n_rows, random_state=RANDOM_STATE)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(out_path, index=False)
    return sample