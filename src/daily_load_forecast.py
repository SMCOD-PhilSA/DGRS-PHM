# Author: Arcee Juan
# DGS Daily Load Forecast (7-Day, 5° Elevation)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, Topos, utc
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

TRACKED_SAT_IDS = [
    "27424",  # AQUA
    "43672",  # DIWATA 2B
    "37849",  # NPP
    "25994",  # TERRA
    "31698",  # TERRA SAR-X
]

GS_LAT = 7.14
GS_LON = 125.65
GS_ALT = 105

MIN_ELEV = 5
HORIZON_DAYS = 7
SAMPLE_INTERVAL = 30  # seconds

# Daily normalization caps (adjustable)
DURATION_CAP_DAY = 15000     # ~4.2 hours
MOTION_CAP_DAY = 50000       # deg

TLE_FILE = Path("tle_latest.txt")
OUTPUT_FILE = Path("daily_load_forecast.csv")


# -------------------------------------------------
# Load Satellites
# -------------------------------------------------

def load_satellites():
    satellites = []
    used_ids = []

    with open(TLE_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(0, len(lines), 3):
        name = lines[i]
        l1 = lines[i+1]
        l2 = lines[i+2]
        norad = l1[2:7].strip()

        if norad in TRACKED_SAT_IDS:
            satellites.append(EarthSatellite(l1, l2, name))
            used_ids.append(norad)

    missing = set(TRACKED_SAT_IDS) - set(used_ids)
    if missing:
        print("WARNING: Missing TLEs for:", missing)

    return satellites


# -------------------------------------------------
# Compute Daily Load
# -------------------------------------------------

def compute_daily_load():

    ts = load.timescale()
    satellites = load_satellites()

    observer = Topos(
        latitude_degrees=GS_LAT,
        longitude_degrees=GS_LON,
        elevation_m=GS_ALT
    )

    today = datetime.utcnow().replace(tzinfo=utc)

    daily_rows = []

    for d in range(HORIZON_DAYS):

        start_dt = (today + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = start_dt + timedelta(days=1)

        t0 = ts.from_datetime(start_dt)
        t1 = ts.from_datetime(end_dt)

        passes = 0
        total_duration = 0
        total_az = 0
        total_el = 0

        for sat in satellites:

            t, events = sat.find_events(observer, t0, t1, altitude_degrees=MIN_ELEV)

            rise_time = None

            for ti, event in zip(t, events):

                if event == 0:
                    rise_time = ti.utc_datetime()

                elif event == 2 and rise_time is not None:

                    set_time = ti.utc_datetime()
                    duration = (set_time - rise_time).total_seconds()

                    if duration > 0:
                        passes += 1
                        total_duration += duration

                        sample_seconds = np.arange(0, duration, SAMPLE_INTERVAL)
                        sample_times = [
                            rise_time + timedelta(seconds=float(s))
                            for s in sample_seconds
                        ]

                        ts_samples = ts.utc(sample_times)

                        difference = sat - observer
                        topo = difference.at(ts_samples)

                        alt, az, _ = topo.altaz()

                        total_az += np.sum(np.abs(np.diff(az.degrees)))
                        total_el += np.sum(np.abs(np.diff(alt.degrees)))

                    rise_time = None

        duration_norm = min(total_duration / DURATION_CAP_DAY, 1)
        motion_norm = min((total_az + total_el) / MOTION_CAP_DAY, 1)

        load_score = 0.6 * duration_norm + 0.4 * motion_norm

        daily_rows.append({
            "date": start_dt.date(),
            "passes": passes,
            "duration_sec": total_duration,
            "az_motion_deg": total_az,
            "el_motion_deg": total_el,
            "load_score_day": load_score
        })

    return pd.DataFrame(daily_rows)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    df = compute_daily_load()
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n==============================")
    print("DGS 7-Day Daily Load Forecast")
    print("==============================")
    print(df)
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()