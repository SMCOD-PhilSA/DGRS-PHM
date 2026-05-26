# Author: Arcee Juan

import os
import re
import csv
from pathlib import Path
from datetime import datetime


# ============================================================
# PROJECT ROOT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data"
OUTPUT_FILE = BASE_DIR / "master_events.csv"


# ============================================================
# REGEX PATTERNS
# ============================================================

TIMESTAMP_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
)

AXIS_PATTERN = re.compile(
    r"(Upper/X east|Lower/Y north)",
    re.IGNORECASE
)

# STRICT error code extraction ONLY if explicitly written
ERROR_CODE_PATTERN = re.compile(
    r"(?:error\s*code[:=\s]+)(\d+)",
    re.IGNORECASE
)

MOTION_PATTERN = re.compile(
    r"\b(Go|Stop|Idle|Blocked|Released|Applying brake|Releasing brake)\b",
    re.IGNORECASE
)


# ============================================================
# EVENT TYPE DETECTION
# ============================================================

def detect_event_type(line: str) -> str:
    lower = line.lower()

    if lower.startswith("fault"):
        return "Fault"
    elif lower.startswith("motion"):
        return "Motion"
    elif lower.startswith("track"):
        return "Track"
    elif lower.startswith("reset"):
        return "Reset"
    else:
        return "Other"


# ============================================================
# LINE PARSER (STRICT)
# ============================================================

def parse_event_line(line: str):

    line = line.strip().replace("\x00", "")
    if not line:
        return None

    ts_match = TIMESTAMP_PATTERN.search(line)
    if not ts_match:
        return None

    ts_str = ts_match.group(1)

    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None

    # Remove timestamp portion
    content = line[len(ts_str) + 1:]  # +1 for comma after timestamp

    event_type = detect_event_type(content)

    axis_match = AXIS_PATTERN.search(content)
    axis = axis_match.group(1) if axis_match else ""

    # STRICT error code extraction
    error_match = ERROR_CODE_PATTERN.search(content)
    error_code = error_match.group(1) if error_match else ""

    motion_match = MOTION_PATTERN.search(content)
    motion_state = motion_match.group(1) if motion_match else ""

    # Only return Fault/Motion/Track/Reset lines from Events
    if event_type in ["Fault", "Motion", "Track", "Reset"]:
        return {
            "timestamp": ts.isoformat(),
            "event_type": event_type,
            "axis": axis,
            "error_code": error_code,
            "motion_state": motion_state,
        }

    return None


# ============================================================
# FILE PROCESSOR
# ============================================================

def process_events_file(filepath: Path, writer):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                parsed = parse_event_line(raw_line)
                if parsed:
                    writer.writerow(parsed)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("Starting STRICT DGS antenna log parsing")
    print(f"Data root: {DATA_ROOT}")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    if not DATA_ROOT.exists():
        print("DATA_ROOT folder not found.")
        return

    total_files = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:

        fieldnames = [
            "timestamp",
            "event_type",
            "axis",
            "error_code",
            "motion_state"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # STRICT directory traversal:
        # data/YYYY/Events or data/YYYY/Metrics
        for year_dir in DATA_ROOT.iterdir():

            if not year_dir.is_dir():
                continue

            events_dir = year_dir / "Events"
            metrics_dir = year_dir / "Metrics"

            # --- PROCESS EVENTS ONLY FOR FAULTS ---
            if events_dir.exists():
                for file in events_dir.iterdir():
                    if file.suffix.lower() in [".txt", ".log"]:
                        print(f"Processing Events: {file}")
                        process_events_file(file, writer)
                        total_files += 1

            # --- DO NOT EXTRACT FAULTS FROM METRICS ---
            if metrics_dir.exists():
                print(f"Skipping Metrics parsing for fault extraction in {metrics_dir}")

    print()
    print(f"Parsing complete. Files processed: {total_files}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()