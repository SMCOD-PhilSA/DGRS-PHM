# ============================================
# Motion Lock Probability Estimator
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT = BASE_DIR / "metrics_servo_stress_daily.csv"

OUTPUT = BASE_DIR / "motion_lock_probability.csv"


def load_data():

    df = pd.read_csv(INPUT, parse_dates=["date"])
    df = df.sort_values("date")

    return df


def compute_features(df):

    stress = df["servo_stress_index"]

    df["stress_slope"] = stress.diff()

    df["stress_volatility"] = stress.rolling(5).std()

    df["instability_score"] = (
        0.5 * stress +
        0.3 * df["stress_slope"].fillna(0) +
        0.2 * df["stress_volatility"].fillna(0)
    )

    return df


def compute_probability(df):

    latest = df.iloc[-1]

    stress = latest["servo_stress_index"]
    slope = latest["stress_slope"]
    vol = latest["stress_volatility"]
    inst = latest["instability_score"]

    # normalize ranges empirically
    stress_factor = stress
    slope_factor = max(slope, 0)
    vol_factor = vol if not np.isnan(vol) else 0
    inst_factor = inst

    risk_score = (
        0.40 * stress_factor +
        0.30 * slope_factor +
        0.15 * vol_factor +
        0.15 * inst_factor
    )

    # convert to probability curve
    p_24 = min(1.0, risk_score * 0.6)
    p_48 = min(1.0, risk_score * 0.9)
    p_72 = min(1.0, risk_score * 1.2)

    return p_24, p_48, p_72


def main():

    print("\n================================")
    print("Motion Lock Probability Estimate")
    print("================================")

    df = load_data()

    df = compute_features(df)

    p24, p48, p72 = compute_probability(df)

    out = pd.DataFrame({
        "window_hours":[24,48,72],
        "motion_lock_probability":[p24,p48,p72]
    })

    out.to_csv(OUTPUT, index=False)

    print("\nMotion Lock Risk:")

    print(f"24 hours : {p24*100:.1f}%")
    print(f"48 hours : {p48*100:.1f}%")
    print(f"72 hours : {p72*100:.1f}%")

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()