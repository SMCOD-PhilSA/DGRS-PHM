# ============================================
# DGS Antenna Health Index Generator
# Author: Arcee Juan
# Proper PHM Model
# ============================================

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "metrics_daily_features.csv"
OUTPUT_FILE = BASE_DIR / "antenna_health_index.csv"


# ============================================
# NORMALIZATION
# ============================================

def normalize(series):

    series = pd.to_numeric(series, errors="coerce").fillna(0)

    lo = series.quantile(0.05)
    hi = series.quantile(0.95)

    if pd.isna(lo) or pd.isna(hi) or hi - lo == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return ((series - lo) / (hi - lo)).clip(0, 1)


# ============================================
# HEALTH INDEX
# ============================================

def compute_health_index():

    print("\nReading daily features...")

    df = pd.read_csv(INPUT_FILE)

    if "date" not in df.columns:
        raise ValueError("metrics_daily_features.csv must contain a 'date' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    required_cols = [
        "tracking_ratio",
        "failed_movement_attempts",
        "failed_movement_rate",
        "error_spikes",
        "error_spike_ratio",
        "Upper axis current_mean",
        "Cabinet temperature_max"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --------------------------------
    # Operational gating
    # --------------------------------
    # More conservative: antenna considered operational when there is
    # at least some tracking activity in the day
    df["is_operational_day"] = (df["tracking_ratio"] >= 0.005).astype(int)

    # --------------------------------
    # Normalize degradation indicators
    # --------------------------------
    # Use rates, not raw counts
    df["failed_rate_norm"] = normalize(df["failed_movement_rate"])
    df["error_rate_norm"] = normalize(df["error_spike_ratio"])
    df["current_norm"] = normalize(df["Upper axis current_mean"])
    df["temp_norm"] = normalize(df["Cabinet temperature_max"])

    # --------------------------------
    # Reduce spike penalty during active tracking
    # because some servo correction is normal
    # --------------------------------
    df["error_rate_adjusted"] = df["error_rate_norm"]

    tracking_mask = df["tracking_ratio"] >= 0.03
    df.loc[tracking_mask, "error_rate_adjusted"] = (
        df.loc[tracking_mask, "error_rate_norm"] * 0.35
    )

    # --------------------------------
    # Health components
    # Higher = healthier
    # Do not let tracking dominate health.
    # Tracking is only a gate for IDLE vs OPERATIONAL.
    # --------------------------------
    movement_score = 1 - df["failed_rate_norm"]
    error_score = 1 - df["error_rate_adjusted"]
    current_score = 1 - df["current_norm"]
    temp_score = 1 - df["temp_norm"]

    df["health_index"] = (
        0.45 * movement_score +
        0.30 * error_score +
        0.15 * current_score +
        0.10 * temp_score
    ).clip(0, 1)

    # --------------------------------
    # Health classification
    # Higher health = better
    # --------------------------------
    conditions = [
        df["is_operational_day"] == 0,
        (df["is_operational_day"] == 1) & (df["health_index"] >= 0.70),
        (df["is_operational_day"] == 1) & (df["health_index"] >= 0.45) & (df["health_index"] < 0.70),
        (df["is_operational_day"] == 1) & (df["health_index"] < 0.45)
    ]

    labels = [
        "IDLE",
        "HEALTHY",
        "WARNING",
        "CRITICAL"
    ]

    df["health_status"] = np.select(conditions, labels, default="UNKNOWN")

    df = df.sort_values("date")
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)


# ============================================
# MAIN
# ============================================

def main():

    compute_health_index()

    print("\nHealth index generated successfully.")


if __name__ == "__main__":
    main()