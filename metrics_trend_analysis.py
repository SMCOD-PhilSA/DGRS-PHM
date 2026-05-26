# ============================================
# DGS Metrics Trend Analysis
# Author: Arcee Juan
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "metrics_master.csv"
OUTPUT_FILE = BASE_DIR / "metrics_daily_trends.csv"

ANALYSIS_DIR = BASE_DIR / "analysis_outputs"
ANALYSIS_DIR.mkdir(exist_ok=True)


# --------------------------------------------
# LOAD DATA
# --------------------------------------------

def load_data():

    print("Loading metrics_master.csv...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    df = df.dropna(subset=["Time"])

    df["date"] = df["Time"].dt.date

    return df


# --------------------------------------------
# BUILD DAILY FEATURES
# --------------------------------------------

def build_daily_features(df):

    print("Computing daily features...")

    metrics = [
        "Upper axis current",
        "Lower axis current",
        "Upper/X following error",
        "Lower/Y following error",
        "Upper/X velocity",
        "Lower/Y velocity",
        "Casting temperature",
        "Cabinet temperature",
        "Ambient temperature"
    ]

    existing = [m for m in metrics if m in df.columns]

    # convert numeric columns safely
    for col in existing:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    daily = df.groupby("date")[existing].agg(["mean","max","std"])

    daily.columns = [f"{c[0]}_{c[1]}" for c in daily.columns]

    daily.reset_index(inplace=True)

    daily.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)

    return daily


# --------------------------------------------
# PLOT AXIS CURRENT TREND
# --------------------------------------------

def plot_axis_current_trend(daily):

    plt.figure(figsize=(12,6))

    if "Upper axis current_mean" in daily.columns:
        plt.plot(daily["date"], daily["Upper axis current_mean"], label="Upper axis current")

    if "Lower axis current_mean" in daily.columns:
        plt.plot(daily["date"], daily["Lower axis current_mean"], label="Lower axis current")

    plt.title("Axis Current Trend Over Time")
    plt.ylabel("Current")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()

    plt.savefig(ANALYSIS_DIR / "axis_current_trend.png")

    plt.close()


# --------------------------------------------
# PLOT FOLLOWING ERROR TREND
# --------------------------------------------

def plot_following_error(daily):

    plt.figure(figsize=(12,6))

    if "Upper/X following error_mean" in daily.columns:
        plt.plot(daily["date"], daily["Upper/X following error_mean"], label="Upper error")

    if "Lower/Y following error_mean" in daily.columns:
        plt.plot(daily["date"], daily["Lower/Y following error_mean"], label="Lower error")

    plt.title("Following Error Trend")
    plt.ylabel("Error")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()

    plt.savefig(ANALYSIS_DIR / "following_error_trend.png")

    plt.close()


# --------------------------------------------
# TEMPERATURE TREND
# --------------------------------------------

def plot_temperature(daily):

    plt.figure(figsize=(12,6))

    if "Cabinet temperature_mean" in daily.columns:
        plt.plot(daily["date"], daily["Cabinet temperature_mean"], label="Cabinet")

    if "Casting temperature_mean" in daily.columns:
        plt.plot(daily["date"], daily["Casting temperature_mean"], label="Casting")

    if "Ambient temperature_mean" in daily.columns:
        plt.plot(daily["date"], daily["Ambient temperature_mean"], label="Ambient")

    plt.title("Temperature Trend")
    plt.ylabel("Temperature")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()

    plt.savefig(ANALYSIS_DIR / "temperature_trend.png")

    plt.close()


# --------------------------------------------
# APRIL-MAY 2023 EVENT ANALYSIS
# --------------------------------------------

def analyze_2023_event(daily):

    daily["date"] = pd.to_datetime(daily["date"])

    subset = daily[
        (daily["date"] >= "2023-03-15") &
        (daily["date"] <= "2023-06-15")
    ]

    if subset.empty:

        print("No data in April–May 2023 window.")

        return

    plt.figure(figsize=(12,6))

    if "Upper axis current_mean" in subset.columns:
        plt.plot(subset["date"], subset["Upper axis current_mean"], label="Upper axis current")

    if "Lower axis current_mean" in subset.columns:
        plt.plot(subset["date"], subset["Lower axis current_mean"], label="Lower axis current")

    plt.title("Axis Current Around April–May 2023 Event")
    plt.ylabel("Current")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()

    plt.savefig(ANALYSIS_DIR / "2023_motion_lock_current.png")

    plt.close()


# --------------------------------------------
# MAIN
# --------------------------------------------

def main():

    print("\n==============================")
    print("Metrics Trend Analysis")
    print("==============================\n")

    df = load_data()

    daily = build_daily_features(df)

    plot_axis_current_trend(daily)

    plot_following_error(daily)

    plot_temperature(daily)

    analyze_2023_event(daily)

    print("\nAnalysis complete.")

    print("Outputs saved to:", ANALYSIS_DIR)


if __name__ == "__main__":
    main()