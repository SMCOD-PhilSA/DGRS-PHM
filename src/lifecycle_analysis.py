# Author: Arcee Juan
# Lifecycle Analysis — Error Code Faults Only

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "master_events.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading master_events.csv...")

    df = pd.read_csv(DATA_FILE)

    print("Converting timestamps...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"])

    return df


# ============================================================
# FILTER STRICT ERROR-CODE FAULTS
# ============================================================

def filter_lower_axis_faults(df):

    # Keep only Fault rows
    df = df[df["event_type"] == "Fault"]

    # Keep only rows with non-empty error codes
    df = df[df["error_code"].notna()]
    df = df[df["error_code"].astype(str).str.strip() != ""]

    # Lower axis only
    df = df[df["axis"].str.contains("Lower", na=False)]

    return df.sort_values("timestamp")


# ============================================================
# DAILY FAULT COUNT
# ============================================================

def compute_daily_faults(df):

    df["date"] = df["timestamp"].dt.date

    daily = df.groupby("date").size().reset_index(name="fault_count")
    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.set_index("date").asfreq("D", fill_value=0)

    daily["rolling_7d"] = daily["fault_count"].rolling(7).mean()

    return daily


# ============================================================
# MTBF
# ============================================================

def compute_mtbf(df):

    df = df.sort_values("timestamp")

    df["delta_seconds"] = df["timestamp"].diff().dt.total_seconds()

    mtbf = df[["timestamp", "delta_seconds"]].dropna()

    return mtbf


# ============================================================
# PLOTS
# ============================================================

def plot_results(daily, mtbf):

    # --- DAILY FAULTS ---
    plt.figure()
    plt.plot(daily.index, daily["fault_count"], label="Daily Faults")
    plt.plot(daily.index, daily["rolling_7d"], label="7-Day Rolling Avg")

    # Key lifecycle markers
    plt.axvline(pd.to_datetime("2023-05-24"), linestyle="--",
                label="Motion Lock 2023-05-24")
    plt.axvline(pd.to_datetime("2024-10-17"), linestyle="--",
                label="Motor Replacement 2024-10-17")
    plt.axvline(pd.to_datetime("2026-01-22"), linestyle="--",
                label="Motor Replacement 2026-01-22")

    plt.title("Lower Axis Daily Faults (Error-Code Only)")
    plt.xlabel("Date")
    plt.ylabel("Fault Count")
    plt.legend()
    plt.show()

    # --- MTBF ---
    plt.figure()
    plt.plot(mtbf["timestamp"], mtbf["delta_seconds"])

    plt.axvline(pd.to_datetime("2023-05-24"), linestyle="--")
    plt.axvline(pd.to_datetime("2024-10-17"), linestyle="--")
    plt.axvline(pd.to_datetime("2026-01-22"), linestyle="--")

    plt.title("Lower Axis MTBF (Seconds)")
    plt.xlabel("Date")
    plt.ylabel("Seconds Between Faults")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df_faults = filter_lower_axis_faults(df)

    print("Total Lower Axis Faults (with error codes):",
          len(df_faults))

    daily = compute_daily_faults(df_faults)
    mtbf = compute_mtbf(df_faults)

    plot_results(daily, mtbf)


if __name__ == "__main__":
    main()