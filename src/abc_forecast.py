# Author: Arcee Juan
# DGS ABCD Forecast System (Health + Load Integrated)

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "daily_health_with_index.csv"
LOAD_FILE = BASE_DIR / "weekly_load_forecast.csv"
OUTPUT_FILE = BASE_DIR / "abc_forecast.csv"

# Forecast horizons
HORIZON_A = 7
HORIZON_B = 7
HORIZON_FAILURE = 30
HORIZON_INTERVENTION = 60

B_MIN_CRIT_DAYS = 4
N_BINS = 8

# Load weights
LOAD_WEIGHT_A = 0.15   # small influence
LOAD_WEIGHT_CD = 0.35  # moderate influence

# Known historical anchors
FAILURE_EVENTS = ["2023-04-05"]
INTERVENTION_EVENTS = ["2024-10-17", "2026-01-22"]


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def health_score(series):
    return (1 - series.clip(0, 100) / 100).clip(0, 1)

def critical_streak(df):
    mask = (df["risk_level"] == "Critical") & (df["active_flag"])
    streak = []
    run = 0
    for v in mask:
        run = run + 1 if v else 0
        streak.append(run)
    return pd.Series(streak)

def future_event(df, event_dates, horizon):
    dates = pd.to_datetime(event_dates)
    dts = pd.to_datetime(df["date"])
    y = np.zeros(len(df))
    for i, d in enumerate(dts):
        end = d + pd.Timedelta(days=horizon)
        y[i] = 1 if any((dates > d) & (dates <= end)) else 0
    return y

def future_any_critical(df, horizon):
    mask = ((df["risk_level"] == "Critical") & (df["active_flag"])).values
    y = np.zeros(len(df))
    for i in range(len(df)):
        end = min(len(df), i + 1 + horizon)
        y[i] = 1 if mask[i+1:end].any() else 0
    return y

def future_remain_critical(df, horizon, min_days):
    mask = ((df["risk_level"] == "Critical") & (df["active_flag"])).values
    y = np.zeros(len(df))
    for i in range(len(df)):
        end = min(len(df), i + 1 + horizon)
        y[i] = 1 if mask[i+1:end].sum() >= min_days else 0
    return y

def calibrate(score, target):
    tmp = pd.DataFrame({"s": score, "y": target}).dropna()
    if tmp["s"].nunique() < 2:
        return [(0, 1, tmp["y"].mean())]

    bins = pd.qcut(tmp["s"], q=N_BINS, duplicates="drop")
    grouped = tmp.groupby(bins, observed=False)

    calib = []
    for _, g in grouped:
        calib.append((g["s"].min(), g["s"].max(), g["y"].mean()))
    return calib

def map_prob(calib, val):
    for lo, hi, p in calib:
        if lo <= val <= hi:
            return float(p)
    return float(calib[-1][2])

def load_score_from_file():
    if not LOAD_FILE.exists():
        print("Load file missing. Using 0.50")
        return 0.50
    df = pd.read_csv(LOAD_FILE)
    return float(np.clip(df["load_score"].iloc[-1], 0, 1))


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    df = pd.read_csv(INPUT_FILE).sort_values("date").reset_index(drop=True)
    df["active_flag"] = df["active_flag"].astype(bool)

    load_score = load_score_from_file()

    df["health_score"] = health_score(df["health_index"])
    df["critical_streak"] = critical_streak(df)
    streak_norm = (df["critical_streak"].clip(0, 14) / 14).clip(0, 1)

    # Targets
    df["yA"] = future_any_critical(df, HORIZON_A)
    df["yB"] = future_remain_critical(df, HORIZON_B, B_MIN_CRIT_DAYS)
    df["yC"] = future_event(df, FAILURE_EVENTS, HORIZON_FAILURE)
    df["yD"] = future_event(df, INTERVENTION_EVENTS, HORIZON_INTERVENTION)

    is_crit = (df["risk_level"] == "Critical") & (df["active_flag"])
    is_noncrit = ~is_crit

    # Scores
    df["score_A"] = (0.85 * df["health_score"] +
                     LOAD_WEIGHT_A * load_score).clip(0, 1)

    df["score_B"] = (0.6 * df["health_score"] +
                     0.4 * streak_norm).clip(0, 1)

    base_cd = (0.75 * df["health_score"] +
               0.25 * streak_norm)

    df["score_CD"] = ((1 - LOAD_WEIGHT_CD) * base_cd +
                      LOAD_WEIGHT_CD * load_score).clip(0, 1)

    # Calibration
    calib_A = calibrate(df.loc[is_noncrit, "score_A"],
                        df.loc[is_noncrit, "yA"])

    calib_B = calibrate(df.loc[is_crit, "score_B"],
                        df.loc[is_crit, "yB"])

    calib_C = calibrate(df["score_CD"], df["yC"])
    calib_D = calibrate(df["score_CD"], df["yD"])

    # Map probabilities
    df["pA"] = df.apply(
        lambda r: map_prob(calib_A, r["score_A"])
        if not ((r["risk_level"] == "Critical") and r["active_flag"])
        else np.nan, axis=1)

    df["pB"] = df.apply(
        lambda r: map_prob(calib_B, r["score_B"])
        if ((r["risk_level"] == "Critical") and r["active_flag"])
        else np.nan, axis=1)

    df["pC"] = df["score_CD"].apply(lambda v: map_prob(calib_C, v))
    df["pD"] = df["score_CD"].apply(lambda v: map_prob(calib_D, v))

    df["load_score_used"] = load_score
    df.to_csv(OUTPUT_FILE, index=False)

    # Latest output
    latest = df.iloc[-1]

    print("\n==============================")
    print("DGS Predictive Risk Forecast")
    print("==============================")
    print(f"Date: {latest['date']}")
    print(f"Risk Level: {latest['risk_level']}")
    print(f"Health Index: {latest['health_index']:.1f}")
    print(f"Load Score: {load_score:.3f}\n")

    if pd.isna(latest["pA"]):
        print("A) Enter Critical (7d): N/A (already Critical)")
    else:
        print(f"A) Enter Critical (7d): {latest['pA']*100:.1f}%")

    if pd.isna(latest["pB"]):
        print("B) Remain Critical (7d): N/A")
    else:
        print(f"B) Remain Critical (7d): {latest['pB']*100:.1f}%")

    print(f"C) Failure (30d): {latest['pC']*100:.1f}%")
    print(f"D) Intervention (60d): {latest['pD']*100:.1f}%")
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()