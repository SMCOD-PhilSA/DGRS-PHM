# ============================================
# DGS Motion Lock Event Analysis
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT = BASE_DIR / "metrics_servo_stress_daily.csv"

OUT_2023 = BASE_DIR / "motionlock_2023_analysis.png"
OUT_RECENT = BASE_DIR / "motionlock_recent_analysis.png"


def load_data():

    df = pd.read_csv(INPUT, parse_dates=["date"])

    df = df.sort_values("date")

    stress = df["servo_stress_index"]

    df["stress_slope"] = stress.diff()
    df["stress_volatility"] = stress.rolling(5).std()

    df["instability_score"] = (
        0.5 * stress +
        0.3 * df["stress_slope"].fillna(0) +
        0.2 * df["stress_volatility"].fillna(0)
    )

    return df


def plot_event(df, start, end, title, outfile):

    event = df[
        (df["date"] >= start) &
        (df["date"] <= end)
    ]

    print("\n===================================")
    print(title)
    print("===================================\n")

    print(event.tail(10)[[
        "date",
        "servo_stress_index",
        "stress_slope",
        "stress_volatility",
        "instability_score"
    ]])

    fig, ax = plt.subplots(3,1, figsize=(12,10), sharex=True)

    ax[0].plot(event["date"], event["servo_stress_index"])
    ax[0].set_title("Servo Stress Index")

    ax[1].plot(event["date"], event["stress_slope"])
    ax[1].set_title("Stress Slope")

    ax[2].plot(event["date"], event["instability_score"])
    ax[2].set_title("Instability Score")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(outfile)
    plt.close()

    print("\nSaved plot:", outfile)


def main():

    df = load_data()

    # April–May 2023 motion lock
    plot_event(
        df,
        "2023-03-01",
        "2023-06-30",
        "APR–MAY 2023 MOTION LOCK",
        OUT_2023
    )

    # Recent motion lock window
    plot_event(
        df,
        "2025-10-01",
        "2026-03-31",
        "RECENT MOTION LOCK",
        OUT_RECENT
    )


if __name__ == "__main__":
    main()