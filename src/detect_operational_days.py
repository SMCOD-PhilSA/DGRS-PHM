import pandas as pd
from pathlib import Path
import re

BASE = Path.cwd()

INPUT = BASE / "metrics_master.csv"
OUTPUT = BASE / "operational_days.csv"

CHUNK = 200000

print("\nScanning antenna movement...\n")

movement_per_day = {}

reader = pd.read_csv(INPUT, chunksize=CHUNK, low_memory=False)

for chunk in reader:

    # convert numeric properly
    for col in [
        "Antenna azimuth velocity",
        "Antenna elevation velocity"
    ]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

    # detect movement
    moving = (
        chunk["Antenna azimuth velocity"].abs() > 0
    ) | (
        chunk["Antenna elevation velocity"].abs() > 0
    )

    chunk["moving"] = moving

    # use row index as fallback grouping
    # (since Time column has no date)
    chunk["day_index"] = chunk.index // 8640

    grouped = chunk.groupby("day_index")["moving"].sum()

    for day, count in grouped.items():
        movement_per_day[day] = movement_per_day.get(day, 0) + int(count)

rows = []

for d in sorted(movement_per_day):
    rows.append({
        "day_index": d,
        "moving_samples": movement_per_day[d],
        "is_operational": movement_per_day[d] > 100
    })

df = pd.DataFrame(rows)

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)

print("\nOperational segments:\n")
print(df[df["is_operational"]])