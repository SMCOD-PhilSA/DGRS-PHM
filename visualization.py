# ============================================
# DGS Predictive Health Monitoring - Advanced Visualization Suite
# Author: Arcee Juan
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STATE_HISTORY = BASE_DIR / "degradation_state_history.csv"
EPISODES = BASE_DIR / "fault_episodes.csv"
DAILY_FEATURES = BASE_DIR / "daily_health_features.csv"

sns.set(style="whitegrid")


# ------------------------------------------------------------
# 1. Degradation State Over Time
# ------------------------------------------------------------

def plot_degradation():
    df = pd.read_csv(STATE_HISTORY, parse_dates=["date"])
    plt.figure(figsize=(10,5))
    plt.plot(df["date"], df["degradation_state"])
    plt.title("Degradation State Over Time")
    plt.ylabel("Degradation (0-100)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 2. Daily Fault Episodes
# ------------------------------------------------------------

def plot_daily_faults():
    df = pd.read_csv(DAILY_FEATURES, parse_dates=["date"])
    plt.figure(figsize=(10,5))
    plt.plot(df["date"], df["total_episode_count"])
    plt.title("Daily Fault Episode Count")
    plt.ylabel("Episodes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 3. Rolling 7-Day Episodes
# ------------------------------------------------------------

def plot_rolling_episodes():
    df = pd.read_csv(DAILY_FEATURES, parse_dates=["date"])
    plt.figure(figsize=(10,5))
    plt.plot(df["date"], df["rolling_7d_episode_count"])
    plt.title("Rolling 7-Day Episode Count")
    plt.ylabel("Rolling Episodes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4. Fault Duration Distribution (Log Scale)
# ------------------------------------------------------------

def plot_duration_distribution():
    df = pd.read_csv(EPISODES)
    plt.figure(figsize=(8,5))
    sns.histplot(df["duration_sec"], bins=60)
    plt.xscale("log")
    plt.title("Fault Duration Distribution (Log Scale)")
    plt.xlabel("Duration (seconds, log scale)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 5. Top Error Codes
# ------------------------------------------------------------

def plot_top_error_codes():
    df = pd.read_csv(EPISODES)
    top_codes = df["error_code"].value_counts().head(10)

    plt.figure(figsize=(8,5))
    top_codes.plot(kind="bar")
    plt.title("Top 10 Error Codes")
    plt.ylabel("Episode Count")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6. Error Code Evolution Over Time
# ------------------------------------------------------------

def plot_error_evolution():
    df = pd.read_csv(EPISODES, parse_dates=["start_time"])
    df["year"] = df["start_time"].dt.year

    top_codes = df["error_code"].value_counts().head(5).index

    filtered = df[df["error_code"].isin(top_codes)]
    pivot = filtered.pivot_table(index="year",
                                 columns="error_code",
                                 values="duration_sec",
                                 aggfunc="count").fillna(0)

    pivot.plot(figsize=(10,5))
    plt.title("Top Error Code Evolution (Yearly)")
    plt.ylabel("Episode Count")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 7. Fault Spike Heatmap (Year x Month)
# ------------------------------------------------------------

def plot_fault_heatmap():
    df = pd.read_csv(EPISODES, parse_dates=["start_time"])
    df["year"] = df["start_time"].dt.year
    df["month"] = df["start_time"].dt.month

    pivot = df.pivot_table(index="year",
                           columns="month",
                           values="duration_sec",
                           aggfunc="count").fillna(0)

    plt.figure(figsize=(10,5))
    sns.heatmap(pivot, cmap="Reds")
    plt.title("Fault Spike Heatmap (Episode Count)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. MTBF Over Time
# ------------------------------------------------------------

def plot_mtbf():
    df = pd.read_csv(EPISODES, parse_dates=["start_time"])
    df = df.sort_values("start_time")

    df["year_month"] = df["start_time"].dt.to_period("M")

    monthly_counts = df.groupby("year_month").size()
    monthly_hours = 30 * 24  # approx monthly operating hours

    mtbf = monthly_hours / monthly_counts
    mtbf.index = mtbf.index.astype(str)

    plt.figure(figsize=(10,5))
    mtbf.plot()
    plt.title("Monthly MTBF (Hours)")
    plt.ylabel("Hours")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 9. Activity vs Degradation Overlay
# ------------------------------------------------------------

def plot_activity_vs_degradation():
    df1 = pd.read_csv(STATE_HISTORY, parse_dates=["date"])
    df2 = pd.read_csv(DAILY_FEATURES, parse_dates=["date"])

    merged = df1.merge(df2, on="date")

    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(merged["date"], merged["degradation_state"])
    ax1.set_ylabel("Degradation")

    ax2 = ax1.twinx()
    ax2.plot(merged["date"], merged["motion_event_count"])
    ax2.set_ylabel("Motion Events")

    plt.title("Activity vs Degradation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 10. Year-over-Year Fault Comparison
# ------------------------------------------------------------

def plot_year_over_year():
    df = pd.read_csv(EPISODES, parse_dates=["start_time"])
    df["year"] = df["start_time"].dt.year

    yearly_counts = df.groupby("year").size()

    plt.figure(figsize=(8,5))
    yearly_counts.plot(kind="bar")
    plt.title("Year-over-Year Fault Episodes")
    plt.ylabel("Episode Count")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    plot_degradation()
    plot_daily_faults()
    plot_rolling_episodes()
    plot_duration_distribution()
    plot_top_error_codes()
    plot_error_evolution()
    plot_fault_heatmap()
    plot_mtbf()
    plot_activity_vs_degradation()
    plot_year_over_year()


if __name__ == "__main__":
    main()