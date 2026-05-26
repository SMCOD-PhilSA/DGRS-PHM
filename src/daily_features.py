# Author: Arcee Juan
# Daily Health Feature Generator (Load-Aware Version)

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress


BASE_DIR = Path(__file__).resolve().parent.parent
EPISODE_FILE = BASE_DIR / "fault_episodes.csv"
MASTER_FILE = BASE_DIR / "master_events.csv"
OUTPUT_FILE = BASE_DIR / "daily_health_features.csv"


def load_data():

    episodes = pd.read_csv(EPISODE_FILE)
    episodes["start_time"] = pd.to_datetime(episodes["start_time"])
    episodes["date"] = episodes["start_time"].dt.date
    episodes["date"] = pd.to_datetime(episodes["date"])

    master = pd.read_csv(MASTER_FILE)
    master["timestamp"] = pd.to_datetime(master["timestamp"], errors="coerce")
    master["date"] = master["timestamp"].dt.date
    master["date"] = pd.to_datetime(master["date"])

    return episodes, master


def build_daily_features(episodes, master):

    # Episode-based features
    daily = episodes.groupby("date").agg(
        total_episode_count=("error_code", "count"),
        total_fault_duration_sec=("duration_sec", "sum"),
        unique_error_codes=("error_code", "nunique")
    ).reset_index()

    # Activity features
    activity = master.groupby("date").agg(
        motion_event_count=("motion_state", lambda x: x.notna().sum()),
        track_event_count=("event_type", lambda x: (x == "Track").sum())
    ).reset_index()

    daily = daily.merge(activity, on="date", how="outer").fillna(0)

    daily = daily.sort_values("date")
    daily = daily.set_index("date").asfreq("D", fill_value=0)

    # Define active flag
    daily["active_flag"] = (
        (daily["motion_event_count"] > 10) |
        (daily["track_event_count"] > 0)
    )

    # Rolling metrics (only for active days)
    daily["rolling_7d_episode_count"] = (
        daily["total_episode_count"]
        .rolling(7)
        .mean()
    )

    daily["rolling_7d_fault_duration"] = (
        daily["total_fault_duration_sec"]
        .rolling(7)
        .mean()
    )

    # Trend slope (14-day window)
    slopes = []

    for i in range(len(daily)):
        if i < 14:
            slopes.append(0)
        else:
            y = daily["total_episode_count"].iloc[i-14:i]
            x = np.arange(len(y))
            slope, _, _, _, _ = linregress(x, y)
            slopes.append(slope)

    daily["episode_trend_slope_14d"] = slopes

    return daily.reset_index()


def main():

    print("Loading episodes and master logs...")
    episodes, master = load_data()

    print("Building daily load-aware features...")
    daily = build_daily_features(episodes, master)

    daily.to_csv(OUTPUT_FILE, index=False)

    print("Daily features updated with activity.")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()