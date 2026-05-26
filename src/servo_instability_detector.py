import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT = BASE_DIR / "metrics_servo_stress_daily.csv"
OUTPUT = BASE_DIR / "servo_instability_events.csv"


def detect_instability(df):

    df = df.sort_values("date")

    stress = df["servo_stress_index"]

    slope = stress.diff()
    volatility = stress.rolling(5).std()
    trend = stress.rolling(3).mean()

    df["stress_slope"] = slope
    df["stress_volatility"] = volatility
    df["stress_trend"] = trend

    # instability score
    df["instability_score"] = (
        0.5 * stress +
        0.3 * slope.fillna(0) +
        0.2 * volatility.fillna(0)
    )

    # dynamic threshold (less strict)
    threshold = df["instability_score"].quantile(0.85)

    df["instability_flag"] = df["instability_score"] > threshold

    return df


def main():

    print("\n==============================")
    print("Servo Instability Detector")
    print("==============================")

    df = pd.read_csv(INPUT, parse_dates=["date"])

    df = detect_instability(df)

    events = df[df["instability_flag"]]

    df.to_csv(OUTPUT, index=False)

    print(f"\nTotal instability events: {len(events)}")

    if len(events) > 0:

        print("\nRecent warnings:\n")

        print(events.tail(10)[[
            "date",
            "servo_stress_index",
            "stress_slope",
            "stress_volatility",
            "instability_score"
        ]])

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()