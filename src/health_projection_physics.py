# ============================================
# DGS PHM - 7-Day Physics Projection (Aligned Dates)
# Author: Arcee Juan
# ============================================

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

STATE_HISTORY_FILE = BASE_DIR / "degradation_state_history.csv"
PARAMS_FILE = BASE_DIR / "physics_params.csv"
DAILY_LOAD_FILE = BASE_DIR / "next_7day_daily_load.csv"

OUT_FORECAST = BASE_DIR / "health_forecast_physics_7day.csv"


def load_params() -> dict:
    df = pd.read_csv(PARAMS_FILE)
    # expects columns: parameter,value
    if "parameter" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{PARAMS_FILE} must have columns: parameter,value")
    return dict(zip(df["parameter"], df["value"]))


def classify_state(d: float) -> str:
    # Same thresholds you’ve been using conceptually
    if d >= 70:
        return "CRITICAL"
    if d >= 60:
        return "DEGRADING"
    if d >= 40:
        return "WATCH"
    return "STABLE"


def main():
    print("\n===================================")
    print("DGS PHM 7-Day Physics Projection")
    print("===================================")

    # --- Load latest degradation state ---
    hist = pd.read_csv(STATE_HISTORY_FILE, parse_dates=["date"]).sort_values("date")
    latest = hist.iloc[-1]
    start_date = latest["date"].normalize()
    start_D = float(latest["degradation_state"])
    start_state = classify_state(start_D)

    # --- Load params ---
    params = load_params()
    damage_scale = float(params.get("damage_scale", 1.5))
    recovery_scale = float(params.get("recovery_scale", 4.5))

    # --- Load daily load forecast ---
    if not DAILY_LOAD_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DAILY_LOAD_FILE}. You must run daily_load_forecast.py (or equivalent) before projection."
        )

    load_df = pd.read_csv(DAILY_LOAD_FILE, parse_dates=["date"]).sort_values("date")

    # REQUIRED column
    if "load_score_day" not in load_df.columns:
        raise ValueError(f"{DAILY_LOAD_FILE} must contain column load_score_day")

    # --- ALIGNMENT FIX: only use days >= start_date ---
    load_df = load_df[load_df["date"] >= start_date].copy()

    if len(load_df) < 7:
        raise RuntimeError(
            f"{DAILY_LOAD_FILE} does not contain 7 days starting from {start_date.date()}.\n"
            f"Regenerate daily load forecast so it includes dates >= {start_date.date()}."
        )

    load_df = load_df.head(7)

    print(f"Starting date: {start_date.date()}")
    print(f"Starting degradation_state: {start_D:.3f}")
    print(f"Starting state: {start_state}")
    print(f"damage_scale: {damage_scale} | recovery_scale: {recovery_scale}")

    # --- Policy: always ACTIVE (no forced REST) ---
    # If you later want “rest rules”, this is the section to modify.
    forecast_rows = []
    D = start_D

    for _, row in load_df.iterrows():
        day = row["date"].date()
        load_score = float(row["load_score_day"])

        # Convert load_score_day into “stress” (0..1)
        stress = max(0.0, min(1.0, load_score))

        # Physics update (bounded)
        damage = damage_scale * stress * (1 - D/100.0)
        recovery = 0.0  # ACTIVE policy = no rest recovery

        D = D + damage - recovery
        D = max(0.0, min(100.0, D))

        forecast_rows.append({
            "date": str(day),
            "load_score_day": round(load_score, 6),
            "action": "ACTIVE",
            "damage": round(damage, 3),
            "recovery": round(recovery, 3),
            "degradation_state": round(D, 3),
            "state": classify_state(D),
        })

    out = pd.DataFrame(forecast_rows)
    print("\n7-Day Forecast:")
    print(out.to_string(index=False))

    out.to_csv(OUT_FORECAST, index=False)
    print(f"\nSaved: {OUT_FORECAST.resolve()}")


if __name__ == "__main__":
    main()