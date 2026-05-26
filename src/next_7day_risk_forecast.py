# ============================================
# DGS Predictive Health Monitoring
# 7-Day Dynamic Risk Forecast
# Integrates Events + Metrics + Degradation
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DEGRADATION = BASE_DIR / "degradation_state_history.csv"
LOAD = BASE_DIR / "next_7day_daily_load.csv"
SERVO = BASE_DIR / "metrics_servo_stress_daily.csv"

OUTPUT = BASE_DIR / "next_7day_risk_forecast.csv"


# ------------------------------
# Load data
# ------------------------------

def load_data():

    deg = pd.read_csv(DEGRADATION, parse_dates=["date"])
    load = pd.read_csv(LOAD, parse_dates=["date"])
    servo = pd.read_csv(SERVO, parse_dates=["date"])

    return deg, load, servo


# ------------------------------
# Compute servo instability
# ------------------------------

def compute_servo_instability(servo):

    servo = servo.sort_values("date")

    stress = servo["servo_stress_index"]

    slope = stress.diff().rolling(3).mean()
    volatility = stress.rolling(5).std()

    instability = (
        0.6 * stress +
        0.2 * slope.fillna(0) +
        0.2 * volatility.fillna(0)
    )

    servo["servo_instability"] = instability.clip(lower=0)

    return servo


# ------------------------------
# Compute base failure rate
# ------------------------------

def estimate_base_failure_rate(deg):

    active_days = len(deg)

    # number of degradation spikes
    spikes = (deg["daily_damage"] > 1.2).sum()

    rate = spikes / active_days

    return max(rate, 0.0005)


# ------------------------------
# Forecast risk
# ------------------------------

def forecast_risk(deg, load, servo):

    last_deg = deg.iloc[-1]
    last_servo = servo.iloc[-1]

    base_rate = estimate_base_failure_rate(deg)

    degradation = last_deg["degradation_state"]
    instability = last_servo["servo_instability"]

    results = []

    for _, row in load.iterrows():

        load_score = row["load_score_day"]

        # degradation multiplier
        deg_factor = 1 + (degradation / 100)

        # servo instability multiplier
        servo_factor = 1 + instability

        # load multiplier
        load_factor = 1 + load_score

        p_fail = base_rate * deg_factor * servo_factor * load_factor
        p_intervene = p_fail * 0.35

        results.append({
            "date": row["date"],
            "load_score_day": load_score,
            "servo_instability": instability,
            "p_failure_day": p_fail,
            "p_intervention_day": p_intervene
        })

    df = pd.DataFrame(results)

    return df


# ------------------------------
# Main
# ------------------------------

def main():

    print("\n========================================")
    print("DGS Dynamic 7-Day Risk Forecast")
    print("Metrics + Events + Physics")
    print("========================================")

    deg, load, servo = load_data()

    servo = compute_servo_instability(servo)

    df = forecast_risk(deg, load, servo)

    df.to_csv(OUTPUT, index=False)

    cumulative_failure = 1 - np.prod(1 - df["p_failure_day"])
    cumulative_intervention = 1 - np.prod(1 - df["p_intervention_day"])

    print("\n7-Day Forecast:\n")
    print(df)

    print("\n----------------------------------------")
    print(f"Cumulative Failure Risk (7d): {cumulative_failure:.2%}")
    print(f"Cumulative Intervention Risk (7d): {cumulative_intervention:.2%}")
    print("----------------------------------------")

    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()