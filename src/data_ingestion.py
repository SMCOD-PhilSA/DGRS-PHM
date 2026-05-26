# ============================================
# DGS Data Ingestion (Events + Metrics)
# Structure: data/YYYY/{Events,Metrics}
# Incremental Processing + Force Parse
# Author: Arcee Juan
# ============================================

import json
import re
import argparse
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

INDEX_FILE = BASE_DIR / "processed_files.json"

EVENT_OUTPUT = BASE_DIR / "master_events.csv"
METRICS_OUTPUT = BASE_DIR / "metrics_master.csv"


IMPORTANT_COLUMNS = [
    "Time",
    "Upper axis current",
    "Lower axis current",
    "Upper/X following error",
    "Lower/Y following error",
    "Upper/X velocity",
    "Lower/Y velocity",
    "Upper/X position",
    "Lower/Y position",
    "Upper/X position command",
    "Lower/Y position command",
    "Antenna azimuth velocity",
    "Antenna elevation velocity",
    "Movement warning",
    "Antenna alarm",
    "Antenna brake",
]


# --------------------------------------------
# Index Handling
# --------------------------------------------

def load_index():

    if INDEX_FILE.exists():
        return set(json.loads(INDEX_FILE.read_text()))

    return set()


def save_index(index):

    INDEX_FILE.write_text(
        json.dumps(sorted(list(index)), indent=2)
    )


# --------------------------------------------
# File Discovery
# --------------------------------------------

def find_event_files():

    return sorted(DATA_DIR.glob("*/Events/*.txt"))


def find_metrics_files():

    return sorted(DATA_DIR.glob("*/Metrics/*.csv"))


# --------------------------------------------
# Event Parsing
# --------------------------------------------

def parse_event_file(path):

    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            if "ERROR" in line or "FAULT" in line:

                rows.append({
                    "source_file": str(path),
                    "event": line.strip()
                })

    if rows:
        return pd.DataFrame(rows)

    return None


# --------------------------------------------
# Metrics Parsing
# --------------------------------------------

def parse_metrics_file(path):

    df = pd.read_csv(
        path,
        skiprows=2,
        usecols=lambda c: c in IMPORTANT_COLUMNS,
        low_memory=False
    )

    if "Time" not in df.columns:
        return None

    # Extract date from filename
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)

    if not match:
        print(f"Could not detect date in filename: {path}")
        return None

    file_date = match.group(1)

    # Convert telemetry time format (MM:SS.s)
    try:

        time_delta = pd.to_timedelta(df["Time"], errors="coerce")

        df["Time"] = pd.to_datetime(file_date) + time_delta

    except Exception as e:

        print(f"Time parsing fallback used for {path}")

        base_time = pd.to_datetime(file_date)

        df["Time"] = base_time + pd.to_timedelta(df.index, unit="s")

    # Metadata
    df["log_date"] = file_date
    df["source_file"] = str(path)

    return df


# --------------------------------------------
# Main
# --------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        help="Force parse a specific file",
        default=None
    )

    args = parser.parse_args()

    processed = load_index()

    event_files = find_event_files()
    metrics_files = find_metrics_files()

    # --------------------------------
    # Force Parse Mode
    # --------------------------------

    if args.force:

        force_path = BASE_DIR / args.force

        if not force_path.exists():
            print(f"File not found: {force_path}")
            return

        print("\n=================================")
        print("FORCE PARSE MODE")
        print("=================================")

        print(f"Parsing {force_path}")

        if force_path.suffix == ".txt":

            df = parse_event_file(force_path)

            if df is not None:

                if EVENT_OUTPUT.exists():
                    df.to_csv(EVENT_OUTPUT, mode="a", header=False, index=False)
                else:
                    df.to_csv(EVENT_OUTPUT, index=False)

            print("Event file parsed.")

        elif force_path.suffix == ".csv":

            df = parse_metrics_file(force_path)

            if df is not None:

                first_write = not METRICS_OUTPUT.exists()

                df.to_csv(
                    METRICS_OUTPUT,
                    mode="a",
                    header=first_write,
                    index=False
                )

            print("Metrics file parsed.")

        else:

            print("Unsupported file type.")

        return


    # --------------------------------
    # Normal Incremental Processing
    # --------------------------------

    new_events = [f for f in event_files if str(f) not in processed]
    new_metrics = [f for f in metrics_files if str(f) not in processed]

    print("\n=================================")
    print("DGS DATA INGESTION")
    print("=================================")

    print(f"New event files: {len(new_events)}")
    print(f"New metrics files: {len(new_metrics)}")


    # --------------------------------
    # Process Events
    # --------------------------------

    event_frames = []

    for f in new_events:

        try:

            df = parse_event_file(f)

            if df is not None:
                event_frames.append(df)

            processed.add(str(f))

            print(f"Parsed EVENT {f}")

        except Exception as e:

            print(f"Failed EVENT {f} | {e}")

    if event_frames:

        df_all = pd.concat(event_frames)

        if EVENT_OUTPUT.exists():
            df_all.to_csv(EVENT_OUTPUT, mode="a", header=False, index=False)
        else:
            df_all.to_csv(EVENT_OUTPUT, index=False)


    # --------------------------------
    # Process Metrics
    # --------------------------------

    first_write = not METRICS_OUTPUT.exists()

    for f in new_metrics:

        try:

            df = parse_metrics_file(f)

            if df is None:
                continue

            df.to_csv(
                METRICS_OUTPUT,
                mode="a",
                header=first_write,
                index=False
            )

            first_write = False

            processed.add(str(f))

            print(f"Parsed METRICS {f}")

        except Exception as e:

            print(f"Failed METRICS {f} | {e}")


    # --------------------------------
    # Save index
    # --------------------------------

    save_index(processed)

    print("\nIngestion complete")


if __name__ == "__main__":
    main()