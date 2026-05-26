# ============================================
# DGS PHM Daily Feature Generator
# Author: Arcee Juan
# Revised PHM Feature Logic
# ============================================

import pandas as pd
import numpy as np
from collections import defaultdict

INPUT_FILE = "metrics_master.csv"
OUT_DAILY = "metrics_daily_features.csv"

CHUNK_SIZE = 200000


NUMERIC_COLS = [
    "Upper axis current",
    "Lower axis current",
    "Upper/X following error",
    "Lower/Y following error",
    "Upper/X velocity",
    "Lower/Y velocity",
    "Antenna azimuth velocity",
    "Antenna elevation velocity",
    "Cabinet temperature",
    "Ambient temperature",
    "Casting temperature"
]


FLAG_COLS = [
    "Movement warning",
    "Upper axis hard limit",
    "Lower axis hard limit",
    "Upper axis e-stop",
    "Lower axis e-stop",
    "Upper axis final limit",
    "Lower axis final limit",
    "Antenna alarm",
    "Antenna brake"
]


# ============================================
# FLAG CONVERSION
# ============================================

def convert_flags(series):

    s = series.astype(str).str.lower().str.strip()

    mapping = {
        "true": 1, "on": 1, "yes": 1, "high": 1, "active": 1,
        "false": 0, "off": 0, "no": 0, "low": 0, "inactive": 0,
        "ok": 0
    }

    return s.map(mapping).fillna(pd.to_numeric(series, errors="coerce")).fillna(0)


# ============================================
# TRACKING DETECTION
# ============================================

def detect_tracking(df):

    needed = ["Antenna brake", "Upper/X velocity", "Lower/Y velocity"]
    for c in needed:
        if c not in df.columns:
            return pd.Series([False] * len(df), index=df.index)

    brake = pd.to_numeric(df["Antenna brake"], errors="coerce").fillna(1)
    vel_x = pd.to_numeric(df["Upper/X velocity"], errors="coerce").fillna(0).abs()
    vel_y = pd.to_numeric(df["Lower/Y velocity"], errors="coerce").fillna(0).abs()

    # Tracking only when brake released and some actual motion exists
    tracking = (brake == 0) & ((vel_x > 0.01) | (vel_y > 0.01))
    return tracking


# ============================================
# FAILED MOVEMENT ATTEMPTS
# Conservative detection
# ============================================

def detect_failed_movements(df):

    needed = ["Antenna brake", "Upper/X velocity", "Lower/Y velocity"]
    for c in needed:
        if c not in df.columns:
            return 0

    brake = pd.to_numeric(df["Antenna brake"], errors="coerce").fillna(1)
    vel_x = pd.to_numeric(df["Upper/X velocity"], errors="coerce").fillna(0).abs()
    vel_y = pd.to_numeric(df["Lower/Y velocity"], errors="coerce").fillna(0).abs()

    brake_prev = brake.shift(1).fillna(brake.iloc[0])

    # Attempt = brake released
    attempt = (brake_prev == 1) & (brake == 0)

    # Fail only if release happened but velocities stayed almost zero
    failed = attempt & (vel_x < 0.005) & (vel_y < 0.005)

    # Collapse consecutive failures into transitions, not raw rows
    failed_edges = failed & (~failed.shift(1).fillna(False))

    return int(failed_edges.sum())


# ============================================
# ERROR SPIKES
# Use physical threshold + ratio
# ============================================

def detect_error_spikes(df):

    if "Upper/X following error" not in df.columns:
        return 0, 0.0

    err = pd.to_numeric(df["Upper/X following error"], errors="coerce").fillna(0).abs()

    # Fixed physical threshold, not daily quantile
    threshold = 0.5

    spikes = (err > threshold).sum()
    ratio = float(spikes) / len(df) if len(df) > 0 else 0.0

    return int(spikes), ratio


# ============================================
# DAILY FEATURE EXTRACTION
# ============================================

def compute_daily_features():

    print("\nReading metrics_master.csv...")

    daily_data = defaultdict(list)

    reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for chunk in reader:

        # ------------------------------
        # TIME HANDLING
        # ------------------------------
        if "Time" in chunk.columns:
            chunk["Time"] = pd.to_datetime(chunk["Time"], errors="coerce")

            if chunk["Time"].notna().any():
                chunk["date"] = chunk["Time"].dt.date
            elif "log_date" in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["log_date"], errors="coerce").dt.date
            else:
                continue

        elif "log_date" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["log_date"], errors="coerce").dt.date
        else:
            continue

        chunk = chunk.dropna(subset=["date"])
        if chunk.empty:
            continue

        # ------------------------------
        # NUMERIC CONVERSION
        # ------------------------------
        for col in NUMERIC_COLS:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # ------------------------------
        # FLAG CONVERSION
        # ------------------------------
        for col in FLAG_COLS:
            if col in chunk.columns:
                chunk[col] = convert_flags(chunk[col])

        grouped = chunk.groupby("date")

        for date, g in grouped:

            if g.empty:
                continue

            record = {"date": date}

            # Numeric stats
            for col in NUMERIC_COLS:
                if col in g.columns:
                    record[col + "_mean"] = g[col].mean()
                    record[col + "_max"] = g[col].max()
                    record[col + "_std"] = g[col].std()

            # Flag counts
            for col in FLAG_COLS:
                if col in g.columns:
                    record[col + "_count"] = g[col].sum()

            # Tracking detection
            tracking_mask = detect_tracking(g)

            record["samples"] = int(len(g))
            record["tracking_samples"] = int(tracking_mask.sum())
            record["tracking_ratio"] = (
                record["tracking_samples"] / record["samples"]
                if record["samples"] > 0 else 0
            )

            # Failed movements
            record["failed_movement_attempts"] = detect_failed_movements(g)
            record["failed_movement_rate"] = (
                record["failed_movement_attempts"] / max(record["samples"], 1)
            )

            # Error spikes
            spikes, spike_ratio = detect_error_spikes(g)
            record["error_spikes"] = spikes
            record["error_spike_ratio"] = spike_ratio

            daily_data[date].append(record)

    rows = []

    for date in sorted(daily_data.keys()):
        df_day = pd.DataFrame(daily_data[date])

        row = {"date": date}

        for c in df_day.columns:
            if c != "date":
                row[c] = df_day[c].mean()

        rows.append(row)

    if len(rows) == 0:
        print("No daily metrics generated.")
        return None

    daily = pd.DataFrame(rows)
    daily = daily.sort_values("date")
    daily.to_csv(OUT_DAILY, index=False)

    print("\nSaved:", OUT_DAILY)
    return daily


# ============================================
# MAIN
# ============================================

def main():

    compute_daily_features()

    print("\nDaily PHM features generated.")


if __name__ == "__main__":
    main()