# ============================================
# DGS Load Forecast - Daily + Weekly
# Author: Arcee Juan
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent

STATE_HISTORY_FILE = BASE_DIR / "degradation_state_history.csv"
OUT_DAILY = BASE_DIR / "next_7day_daily_load.csv"
OUT_WEEKLY = BASE_DIR / "weekly_load_forecast.csv"


def main():
    print("\n==============================")
    print("DGS 7-Day Load Forecast")
    print("==============================")

    # Use latest degradation date as anchor
    hist = pd.read_csv(STATE_HISTORY_FILE, parse_dates=["date"])
    latest_date = hist["date"].max().normalize()

    # For now: simulate daily load score using weekly average logic
    # Replace this later with real orbit pass-based daily computation

    np.random.seed(42)  # deterministic reproducibility

    future_dates = [latest_date + timedelta(days=i) for i in range(1, 8)]

    # Simulate realistic daily load around 0.45 ± small variation
    daily_scores = np.clip(
        np.random.normal(loc=0.47, scale=0.03, size=7),
        0.35,
        0.60
    )

    daily_df = pd.DataFrame({
        "date": future_dates,
        "load_score_day": np.round(daily_scores, 6)
    })

    daily_df.to_csv(OUT_DAILY, index=False)

    # Weekly summary
    weekly_summary = {
        "forecast_generated_utc": pd.Timestamp.utcnow(),
        "average_load_score": float(np.mean(daily_scores)),
        "max_load_score": float(np.max(daily_scores)),
        "min_load_score": float(np.min(daily_scores))
    }

    pd.DataFrame([weekly_summary]).to_csv(OUT_WEEKLY, index=False)

    print("Daily forecast saved:", OUT_DAILY.resolve())
    print("Weekly summary saved:", OUT_WEEKLY.resolve())


if __name__ == "__main__":
    main()