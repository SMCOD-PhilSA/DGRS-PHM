# ============================================
# DGS Antenna PHM Monitor
# Author: Arcee Juan
# Allows testing any date
# ============================================

from pathlib import Path
import pandas as pd
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
FORECAST_FILE = BASE_DIR / "phm_risk_forecast.csv"


def load_data():

    df = pd.read_csv(FORECAST_FILE)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.sort_values("date")

    return df


def get_date_row(df, target_date):

    if target_date is None:
        return df.iloc[-1]

    target_date = pd.to_datetime(target_date).date()

    row = df[df["date"] == target_date]

    if row.empty:
        print(f"\nNo data found for {target_date}")
        print("Closest available dates:")

        print(df.tail(10)[["date"]])

        sys.exit()

    return row.iloc[0]


def print_report(row):

    print("\n=======================================")
    print("DGS ANTENNA PHM STATUS")
    print("=======================================")

    print(f"\nDate: {row['date']}")

    print("\nAntenna Health")
    print(f"Health Index: {round(float(row['health_index']),3)}")
    print(f"Status: {row['health_status']}")

    print("\nMotion Lock Risk")
    print(f"Probability: {round(float(row['motion_lock_probability']),3)}")
    print(f"Risk Level: {row['risk_level']}")

    print("\nRisk Forecast")
    print(f"24h risk: {round(float(row['risk_24h']),3)}")
    print(f"48h risk: {round(float(row['risk_48h']),3)}")
    print(f"72h risk: {round(float(row['risk_72h']),3)}")

    print("\nOperational Indicators")

    if "tracking_ratio" in row:
        print(f"Tracking ratio: {round(float(row['tracking_ratio']),3)}")

    if "failed_movement_attempts" in row:
        print(f"Failed movement attempts: {int(row['failed_movement_attempts'])}")

    if "error_spikes" in row:
        print(f"Error spikes: {int(row['error_spikes'])}")

    print("\n=======================================\n")


def main():

    df = load_data()

    # Optional date argument
    if len(sys.argv) > 1:
        test_date = sys.argv[1]
    else:
        test_date = None

    row = get_date_row(df, test_date)

    print_report(row)


if __name__ == "__main__":
    main()