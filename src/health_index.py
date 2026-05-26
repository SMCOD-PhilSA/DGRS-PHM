# Author: Arcee Juan
# Health Index Calculator (Window-Based, Load-Aware, Smoothed)

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "daily_health_features.csv"
OUTPUT_FILE = BASE_DIR / "daily_health_with_index.csv"


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def robust_normalize(series):
    """
    Normalize using 95th percentile to reduce outlier dominance.
    """
    series = series.fillna(0)
    p95 = np.percentile(series, 95)

    if p95 == 0:
        return series * 0

    return (series / p95).clip(0, 1)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading daily health features...")
    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values("date").reset_index(drop=True)

    # Normalize risk drivers
    df["risk_episodes"] = robust_normalize(df["rolling_7d_episode_count"])
    df["risk_duration"] = robust_normalize(df["rolling_7d_fault_duration"])
    df["risk_trend"] = robust_normalize(abs(df["episode_trend_slope_14d"]))

    # Weighted combined risk
    df["combined_risk"] = (
        0.5 * df["risk_episodes"] +
        0.3 * df["risk_duration"] +
        0.2 * df["risk_trend"]
    ).clip(0, 1)

    # Raw health (window-based only)
    df["health_index_raw"] = (1 - df["combined_risk"]) * 100

    # Smooth using 3-day rolling mean
    df["health_index"] = (
        df["health_index_raw"]
        .rolling(3, min_periods=1)
        .mean()
    )

    # Risk classification
    def classify(row):
        if not row["active_flag"]:
            return "Inactive"
        if row["health_index"] >= 80:
            return "Normal"
        elif row["health_index"] >= 60:
            return "Watch"
        elif row["health_index"] >= 40:
            return "Warning"
        else:
            return "Critical"

    df["risk_level"] = df.apply(classify, axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Window-based health index generated.")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()