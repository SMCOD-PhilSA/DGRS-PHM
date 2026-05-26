# ============================================
# DGS Predictive Health Monitoring
# Recursive 7-Day ML Forecast
# Author: Arcee Juan
# ============================================

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta


# ==============================
# CONFIG
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "ml_model_7day.joblib"
FEATURE_FILE = BASE_DIR / "daily_health_features.csv"

FEATURE_COLS = [
    "total_episode_count",
    "rolling_7d_episode_count",
    "total_fault_duration_sec",
    "episode_trend_slope_14d",
    "unique_error_codes"
]

FORECAST_DAYS = 7


# ==============================
# Forecast Logic
# ==============================

def main():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    latest_row = df.iloc[-1].copy()

    forecast_rows = []

    current_features = latest_row.copy()

    print("\n===================================")
    print("ML 7-Day Degradation Probability Forecast")
    print("===================================")
    print("Starting from:", current_features["date"].date())

    for step in range(1, FORECAST_DAYS + 1):

        X = pd.DataFrame([current_features[FEATURE_COLS].to_dict()])
        p = float(model.predict_proba(X)[0, 1])

        forecast_date = current_features["date"] + timedelta(days=1)

        forecast_rows.append({
            "date": forecast_date,
            "p_degrading": p
        })

        # --- Simple forward simulation logic ---
        # If high probability, simulate slight worsening
        if p > 0.5:
            current_features["rolling_7d_episode_count"] += 1
            current_features["total_episode_count"] += 1
            current_features["total_fault_duration_sec"] *= 1.05
        else:
            current_features["rolling_7d_episode_count"] *= 0.95

        current_features["date"] = forecast_date

    forecast_df = pd.DataFrame(forecast_rows)

    print("\nNext 7 Days Forecast:\n")
    for _, row in forecast_df.iterrows():
        print(f"{row['date'].date()} → {row['p_degrading']:.3f}")

    forecast_df.to_csv(BASE_DIR / "ml_next7day_forecast.csv", index=False)
    print("\nSaved forecast to ml_next7day_forecast.csv")


if __name__ == "__main__":
    main()