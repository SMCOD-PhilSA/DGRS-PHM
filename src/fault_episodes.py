# Author: Arcee Juan

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "master_events.csv"
OUTPUT_FILE = BASE_DIR / "fault_episodes.csv"

EPISODE_GAP_THRESHOLD = 60  # seconds


def load_fault_data():
    print("Loading master_events.csv...")

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df = df[df["event_type"] == "Fault"]

    df = df[df["error_code"].notna()]
    df = df[df["error_code"].astype(str).str.strip() != ""]

    if "axis" not in df.columns:
        df["axis"] = "Unknown"

    df["axis"] = df["axis"].fillna("Unknown")

    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Filtered coded fault log rows: {len(df)}")
    return df


def build_episodes(df):
    episodes = []

    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    for axis in df["axis"].unique():

        df_axis = df[df["axis"] == axis].copy()
        df_axis = df_axis.sort_values("timestamp").reset_index(drop=True)

        current_code = None
        start_time = None
        last_time = None
        log_count = 0

        for _, row in df_axis.iterrows():

            ts = row["timestamp"]
            code = str(row["error_code"])

            if current_code is None:
                current_code = code
                start_time = ts
                last_time = ts
                log_count = 1
                continue

            time_gap = (ts - last_time).total_seconds()

            if code == current_code and time_gap <= EPISODE_GAP_THRESHOLD:
                last_time = ts
                log_count += 1
                continue

            duration = (last_time - start_time).total_seconds()

            if duration > 0:
                episodes.append({
                    "axis": axis,
                    "error_code": current_code,
                    "start_time": start_time,
                    "end_time": last_time,
                    "duration_sec": duration,
                    "log_entries": log_count
                })

            current_code = code
            start_time = ts
            last_time = ts
            log_count = 1

        if current_code is not None:
            duration = (last_time - start_time).total_seconds()

            if duration > 0:
                episodes.append({
                    "axis": axis,
                    "error_code": current_code,
                    "start_time": start_time,
                    "end_time": last_time,
                    "duration_sec": duration,
                    "log_entries": log_count
                })

    return pd.DataFrame(episodes)


def main():
    df = load_fault_data()

    print("Building fault episodes per axis...")
    episodes_df = build_episodes(df)

    print(f"Total fault episodes: {len(episodes_df)}")

    if not episodes_df.empty:
        print("Saving fault_episodes.csv...")
        episodes_df.to_csv(OUTPUT_FILE, index=False)

        print("\nEpisode duration summary (seconds):")
        print(episodes_df["duration_sec"].describe())

        print("\nTop error codes by episode count:")
        print(episodes_df["error_code"].value_counts().head(10))
    else:
        print("No coded fault episodes found.")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()