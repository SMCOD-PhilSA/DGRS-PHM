# ============================================
# Rebuild Master CSV Order
# Ensures chronological ordering of datasets
# Author: Arcee Juan
# ============================================

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

EVENTS_FILE = BASE_DIR / "master_events.csv"
METRICS_FILE = BASE_DIR / "metrics_master.csv"


# --------------------------------------------
# Rebuild Events Order
# --------------------------------------------

def rebuild_events():

    if not EVENTS_FILE.exists():
        print("No events master file found.")
        return

    print("Reordering events...")

    try:

        df = pd.read_csv(EVENTS_FILE, low_memory=False)

        # Attempt to extract timestamp from event text if needed
        if "event" in df.columns:

            df["timestamp"] = df["event"].str.extract(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
            )

            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="coerce"
            )

            df = df.sort_values("timestamp")

        df.to_csv(EVENTS_FILE, index=False)

        print("Events reordered.")

    except Exception as e:

        print("Failed to reorder events:", e)


# --------------------------------------------
# Rebuild Metrics Order
# --------------------------------------------

def rebuild_metrics():

    if not METRICS_FILE.exists():
        print("No metrics master file found.")
        return

    print("Reordering metrics...")

    try:

        df = pd.read_csv(
            METRICS_FILE,
            low_memory=False
        )

        if "Time" in df.columns:

            df["Time"] = pd.to_datetime(
                df["Time"],
                errors="coerce"
            )

            df = df.sort_values("Time")

        df.to_csv(METRICS_FILE, index=False)

        print("Metrics reordered.")

    except Exception as e:

        print("Failed to reorder metrics:", e)


# --------------------------------------------
# Main
# --------------------------------------------

def main():

    print("\n=================================")
    print("REBUILD MASTER ORDER")
    print("=================================")

    rebuild_events()

    rebuild_metrics()

    print("\nMaster datasets successfully reordered.")


if __name__ == "__main__":
    main()