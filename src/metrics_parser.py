# ============================================
# DGS Metrics Parser (Memory Safe)
# Author: Arcee Juan
# FIXED: Proper Antenna Brake Parsing
# ============================================

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

OUTPUT_FILE = BASE_DIR / "metrics_master.csv"


# ============================================
# IMPORTANT TELEMETRY COLUMNS
# ============================================

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
    "Upper axis hard limit",
    "Lower axis hard limit",
    "Upper axis final limit",
    "Lower axis final limit",
    "Upper axis e-stop",
    "Lower axis e-stop",
    "Movement warning",
    "Antenna alarm",
    "Antenna brake",
    "Casting temperature",
    "Cabinet temperature",
    "Ambient temperature",
]


# ============================================
# BRAKE PARSER
# ============================================

def parse_brake(series):
    """
    Convert Antenna brake state to numeric.

    Low  = brake released (antenna allowed to move) → 1
    High = brake engaged  (antenna locked)          → 0
    """

    s = series.astype(str).str.strip().str.lower()

    mapping = {
        "low": 1,
        "high": 0
    }

    return s.map(mapping).fillna(0).astype(int)


# ============================================
# FIND METRICS FILES
# ============================================

def find_metrics_files():

    files = sorted(DATA_DIR.glob("*/Metrics/*.csv"))

    print(f"Found {len(files)} metrics files")

    return files


# ============================================
# PROCESS METRICS FILES
# ============================================

def process_files():

    files = find_metrics_files()

    # remove previous output
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    first_write = True

    for f in files:

        try:

            df = pd.read_csv(
                f,
                skiprows=2,
                usecols=lambda c: c in IMPORTANT_COLUMNS,
                low_memory=False
            )

            if "Time" not in df.columns:
                print(f"Skipped {f.name} (no Time column)")
                continue


            # ----------------------------------------
            # FIX BRAKE PARSING
            # ----------------------------------------

            if "Antenna brake" in df.columns:

                df["Antenna brake raw"] = df["Antenna brake"]

                df["Antenna brake"] = parse_brake(df["Antenna brake"])


            # ----------------------------------------
            # TIME PROCESSING
            # ----------------------------------------

            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

            df = df.dropna(subset=["Time"])

            df = df.sort_values("Time")


            # ----------------------------------------
            # APPEND TO MASTER FILE
            # ----------------------------------------

            df.to_csv(
                OUTPUT_FILE,
                mode="a",
                header=first_write,
                index=False
            )

            first_write = False

            print(f"Processed: {f.name} | rows: {len(df)}")


        except Exception as e:

            print(f"Skipped {f.name} | {e}")


# ============================================
# MAIN
# ============================================

def main():

    print("\n==============================")
    print("DGS Metrics Parser")
    print("==============================\n")

    process_files()

    print("\nMetrics parsing complete")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()