# ============================================
# Motion Lock Prediction (ML)
# ============================================

import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

STRESS_FILE = BASE_DIR / "metrics_servo_stress_daily.csv"
MODEL_FILE = BASE_DIR / "motion_lock_model.pkl"


def load_latest():

    df = pd.read_csv(STRESS_FILE, parse_dates=["date"])

    df = df.sort_values("date")

    stress = df["servo_stress_index"]

    df["stress_slope"] = stress.diff()
    df["stress_volatility"] = stress.rolling(5).std()

    df["instability_score"] = (
        0.5 * stress +
        0.3 * df["stress_slope"].fillna(0) +
        0.2 * df["stress_volatility"].fillna(0)
    )

    df = df.fillna(0)

    return df.iloc[-1]


def main():

    print("\n================================")
    print("ML Motion Lock Prediction")
    print("================================")

    model = joblib.load(MODEL_FILE)

    latest = load_latest()

    X = [[
        latest["servo_stress_index"],
        latest["stress_slope"],
        latest["stress_volatility"],
        latest["instability_score"]
    ]]

    p = model.predict_proba(X)[0][1]

    p24 = min(1.0, p * 0.5)
    p48 = min(1.0, p * 0.8)
    p72 = min(1.0, p * 1.0)

    print("\nMotion Lock Probability:")

    print(f"24h: {p24*100:.1f}%")
    print(f"48h: {p48*100:.1f}%")
    print(f"72h: {p72*100:.1f}%")


if __name__ == "__main__":
    main()