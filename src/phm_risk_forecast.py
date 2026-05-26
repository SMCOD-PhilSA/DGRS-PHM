# ============================================
# DGS Motion Lock Risk Forecast
# Author: Arcee Juan
# Proper PHM Risk Model
# ============================================

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "antenna_health_index.csv"
OUTPUT_FILE = BASE_DIR / "phm_risk_forecast.csv"


def normalize(series):

    series = pd.to_numeric(series, errors="coerce").fillna(0)

    lo = series.quantile(0.05)
    hi = series.quantile(0.95)

    if pd.isna(lo) or pd.isna(hi) or hi - lo == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return ((series - lo) / (hi - lo)).clip(0, 1)


def compute_risk_forecast():

    print("\nReading antenna health index...")

    df = pd.read_csv(INPUT_FILE)

    if "date" not in df.columns:
        raise ValueError("antenna_health_index.csv must contain a 'date' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.sort_values("date").reset_index(drop=True)

    required_cols = [
        "health_index",
        "tracking_ratio",
        "failed_movement_attempts",
        "failed_movement_rate",
        "error_spike_ratio",
        "health_status"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    for col in ["health_index", "tracking_ratio", "failed_movement_attempts", "failed_movement_rate", "error_spike_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --------------------------------
    # Rolling trends
    # --------------------------------
    df["health_3day_avg"] = df["health_index"].rolling(3, min_periods=1).mean()
    df["health_7day_avg"] = df["health_index"].rolling(7, min_periods=1).mean()
    df["health_drop_raw"] = (df["health_7day_avg"] - df["health_3day_avg"]).clip(lower=0)

    df["failed_3day_avg"] = df["failed_movement_rate"].rolling(3, min_periods=1).mean()
    df["failed_7day_avg"] = df["failed_movement_rate"].rolling(7, min_periods=1).mean()
    df["failed_trend_raw"] = (df["failed_3day_avg"] - df["failed_7day_avg"]).clip(lower=0)

    df["error_3day_avg"] = df["error_spike_ratio"].rolling(3, min_periods=1).mean()
    df["error_7day_avg"] = df["error_spike_ratio"].rolling(7, min_periods=1).mean()
    df["error_trend_raw"] = (df["error_3day_avg"] - df["error_7day_avg"]).clip(lower=0)

    df["health_drop"] = normalize(df["health_drop_raw"])
    df["failed_trend"] = normalize(df["failed_trend_raw"])
    df["error_trend"] = normalize(df["error_trend_raw"])

    # --------------------------------
    # Base motion-lock probability
    # --------------------------------
    base_risk = (
        0.45 * (1 - df["health_index"]) +
        0.30 * df["failed_trend"] +
        0.15 * df["health_drop"] +
        0.10 * df["error_trend"]
    ).clip(0, 1)

    df["motion_lock_probability"] = base_risk

    # --------------------------------
    # Idle handling
    # --------------------------------
    idle_mask = df["health_status"].astype(str).str.upper().eq("IDLE")
    df.loc[idle_mask, "motion_lock_probability"] = df.loc[idle_mask, "motion_lock_probability"] * 0.4

    # --------------------------------
    # Risk levels
    # --------------------------------
    conditions = [
        idle_mask,
        df["motion_lock_probability"] < 0.30,
        (df["motion_lock_probability"] >= 0.30) & (df["motion_lock_probability"] < 0.60),
        df["motion_lock_probability"] >= 0.60
    ]

    labels = [
        "UNKNOWN",
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    df["risk_level"] = np.select(conditions, labels, default="UNKNOWN")

    # --------------------------------
    # Forecast horizons
    # --------------------------------
    df["risk_24h"] = (
        df["motion_lock_probability"] +
        0.15 * df["health_drop"] +
        0.10 * df["failed_trend"]
    ).clip(0, 1)

    df["risk_48h"] = (
        df["motion_lock_probability"] +
        0.25 * df["health_drop"] +
        0.20 * df["failed_trend"]
    ).clip(0, 1)

    df["risk_72h"] = (
        df["motion_lock_probability"] +
        0.35 * df["health_drop"] +
        0.25 * df["failed_trend"] +
        0.10 * df["error_trend"]
    ).clip(0, 1)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)


def main():

    compute_risk_forecast()

    print("\nMotion lock risk forecast generated.")


if __name__ == "__main__":
    main()