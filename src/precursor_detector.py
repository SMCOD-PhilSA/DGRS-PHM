import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------
# PATHS
# ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SSI_FILE = PROJECT_ROOT / "metrics_servo_stress_daily.csv"
FAULT_FILE = PROJECT_ROOT / "fault_episodes.csv"

OUTPUT_EVENTS = PROJECT_ROOT / "precursor_events.csv"
OUTPUT_PLOT = PROJECT_ROOT / "precursor_timeline.png"


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

def load_data():

    print("Loading datasets...")

    ssi = pd.read_csv(SSI_FILE)
    faults = pd.read_csv(FAULT_FILE)

    ssi["date"] = pd.to_datetime(ssi["date"])

    faults["start_time"] = pd.to_datetime(faults["start_time"])
    faults["date"] = faults["start_time"].dt.date

    return ssi, faults


# ------------------------------------------------
# DETECT PRECURSORS
# ------------------------------------------------

def detect_precursors(ssi):

    print("Detecting abnormal stress ramps...")

    ssi = ssi.sort_values("date")

    ssi["ssi_diff"] = ssi["servo_stress_index"].diff()

    ssi["ssi_ma"] = ssi["servo_stress_index"].rolling(7).mean()

    ssi["trend_5day"] = ssi["servo_stress_index"].rolling(5).mean()
    ssi["trend_10day"] = ssi["servo_stress_index"].rolling(10).mean()

    ssi["precursor_flag"] = (
        (ssi["trend_5day"] > ssi["trend_10day"]) &
        (ssi["servo_stress_index"] > 0.35)
    )

    events = ssi[ssi["precursor_flag"]]

    return events


# ------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------

def save_precursors(events):

    events.to_csv(OUTPUT_EVENTS, index=False)

    print("Saved:", OUTPUT_EVENTS)


# ------------------------------------------------
# PLOT RESULTS
# ------------------------------------------------

def plot_precursors(ssi, events):

    plt.figure(figsize=(12,6))

    plt.plot(ssi["date"], ssi["servo_stress_index"], label="Servo Stress Index")

    plt.scatter(
        events["date"],
        events["servo_stress_index"],
        color="red",
        label="Precursor"
    )

    plt.title("Servo Stress Precursors")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT)

    plt.close()

    print("Saved:", OUTPUT_PLOT)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    ssi, faults = load_data()

    events = detect_precursors(ssi)

    save_precursors(events)

    plot_precursors(ssi, events)

    print("\nPrecursor detection complete.")


if __name__ == "__main__":
    main()