# ============================================
# Servo Oscillation Detector (FIXED + GATED)
# Author: Arcee Juan
# File: src/servo_oscillation_detector.py
#
# Reads:    metrics_master.csv (project root)
# Outputs:  servo_oscillation_daily.csv (project root)
#           servo_oscillation_events.csv (project root)
#           oscillation_trend.png, oscillation_2023_window.png, oscillation_2026_window.png
# ============================================

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
INPUT_FILE = BASE_DIR / "metrics_master.csv"

OUT_DAILY = BASE_DIR / "servo_oscillation_daily.csv"
OUT_EVENTS = BASE_DIR / "servo_oscillation_events.csv"

PLOT_TREND = BASE_DIR / "oscillation_trend.png"
PLOT_2023 = BASE_DIR / "oscillation_2023_window.png"
PLOT_2026 = BASE_DIR / "oscillation_2026_window.png"

# ----------------------------
# Chunking
# ----------------------------
CHUNK_SIZE = 150_000  # adjust if needed

# ----------------------------
# Columns to use
# ----------------------------
TIME_COL = "Time"

COLS_NUM = [
    "Upper/X following error",
    "Lower/Y following error",
    "Upper axis current",
    "Lower axis current",
    "Upper/X velocity",
    "Lower/Y velocity",
    "Antenna azimuth velocity",
    "Antenna elevation velocity",
]

# Optional flags if present
COLS_FLAGS = [
    "Movement warning",
    "Antenna alarm",
    "Antenna brake",
]

# ----------------------------
# Activity gating thresholds
# ----------------------------
VEL_EPS = 0.001  # minimal movement
MIN_ACTIVE_ROWS_PER_DAY = 200  # days with too few active samples are treated as non-operational
SPIKE_Z = 3.0  # spike when abs(x - median) > SPIKE_Z * MAD


# ============================
# Helpers
# ============================

def to_numeric(series: pd.Series) -> pd.Series:
    """
    Robust conversion:
    - Handles strings like 'High', 'Low', 'True', 'False'
    - Coerces garbage concatenated strings to NaN
    """
    s = series.astype(str).str.strip().str.lower()

    mapping = {
        "high": "1",
        "on": "1",
        "true": "1",
        "yes": "1",
        "active": "1",
        "low": "0",
        "off": "0",
        "false": "0",
        "no": "0",
        "inactive": "0",
        "nan": "",
        "none": "",
    }
    s = s.replace(mapping)
    return pd.to_numeric(s, errors="coerce")


def mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_spike_count(x: np.ndarray, z: float = SPIKE_Z) -> int:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 0
    med = np.median(x)
    m = mad(x)
    if not np.isfinite(m) or m <= 1e-12:
        return 0
    return int(np.sum(np.abs(x - med) > z * m))


def is_active_rows(df: pd.DataFrame) -> pd.Series:
    """
    Activity mask:
    If any velocity channel indicates movement, we treat row as active.
    """
    vcols = [c for c in [
        "Upper/X velocity",
        "Lower/Y velocity",
        "Antenna azimuth velocity",
        "Antenna elevation velocity",
    ] if c in df.columns]

    if not vcols:
        return pd.Series([True] * len(df), index=df.index)

    v = df[vcols].abs().max(axis=1)
    return v > VEL_EPS


# ============================
# Main computation
# ============================

def compute_daily_oscillation() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}")

    print("\n==============================")
    print("Servo Oscillation Detector (FIXED)")
    print("==============================\n")
    print("Reading metrics_master.csv in chunks...")

    acc = defaultdict(lambda: {
        "active_rows": 0,
        "total_rows": 0,

        "upper_err_sum": 0.0, "upper_err_sumsq": 0.0, "upper_err_n": 0,
        "lower_err_sum": 0.0, "lower_err_sumsq": 0.0, "lower_err_n": 0,

        "upper_vel_sum": 0.0, "upper_vel_sumsq": 0.0, "upper_vel_n": 0,
        "lower_vel_sum": 0.0, "lower_vel_sumsq": 0.0, "lower_vel_n": 0,

        "upper_err_spikes": 0,
        "lower_err_spikes": 0,
        "upper_cur_spikes": 0,
        "lower_cur_spikes": 0,
    })

    usecols = [TIME_COL] + [c for c in (COLS_NUM + COLS_FLAGS) if c != TIME_COL]
    usecols = list(dict.fromkeys(usecols))

    reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        usecols=lambda c: c in usecols
    )

    for chunk_i, chunk in enumerate(reader, start=1):
        if TIME_COL not in chunk.columns:
            raise KeyError(f"'{TIME_COL}' column not found in metrics_master.csv")

        chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce")
        chunk = chunk.dropna(subset=[TIME_COL])
        if chunk.empty:
            continue

        chunk["date"] = chunk[TIME_COL].dt.date

        for c in COLS_NUM:
            if c in chunk.columns:
                chunk[c] = to_numeric(chunk[c])

        chunk["active"] = is_active_rows(chunk)

        for day, g in chunk.groupby("date"):
            a = acc[day]
            a["total_rows"] += int(len(g))

            g_active = g[g["active"]].copy()
            a["active_rows"] += int(len(g_active))
            if len(g_active) == 0:
                continue

            ue = g_active["Upper/X following error"].to_numpy() if "Upper/X following error" in g_active.columns else np.array([])
            le = g_active["Lower/Y following error"].to_numpy() if "Lower/Y following error" in g_active.columns else np.array([])
            uc = g_active["Upper axis current"].to_numpy() if "Upper axis current" in g_active.columns else np.array([])
            lc = g_active["Lower axis current"].to_numpy() if "Lower axis current" in g_active.columns else np.array([])
            uv = g_active["Upper/X velocity"].to_numpy() if "Upper/X velocity" in g_active.columns else np.array([])
            lv = g_active["Lower/Y velocity"].to_numpy() if "Lower/Y velocity" in g_active.columns else np.array([])

            def add_stats(x: np.ndarray, sumk: str, sumsqk: str, nk: str):
                x = x[np.isfinite(x)]
                if len(x) == 0:
                    return
                a[sumk] += float(np.sum(x))
                a[sumsqk] += float(np.sum(x * x))
                a[nk] += int(len(x))

            add_stats(ue, "upper_err_sum", "upper_err_sumsq", "upper_err_n")
            add_stats(le, "lower_err_sum", "lower_err_sumsq", "lower_err_n")
            add_stats(uv, "upper_vel_sum", "upper_vel_sumsq", "upper_vel_n")
            add_stats(lv, "lower_vel_sum", "lower_vel_sumsq", "lower_vel_n")

            a["upper_err_spikes"] += robust_spike_count(ue, SPIKE_Z)
            a["lower_err_spikes"] += robust_spike_count(le, SPIKE_Z)
            a["upper_cur_spikes"] += robust_spike_count(uc, SPIKE_Z)
            a["lower_cur_spikes"] += robust_spike_count(lc, SPIKE_Z)

        if chunk_i % 10 == 0:
            print(f"  processed chunk {chunk_i}")

    rows = []
    for day in sorted(acc.keys()):
        a = acc[day]

        def calc_std(sumv, sumsqv, n):
            if n <= 1:
                return np.nan
            mean = sumv / n
            var = (sumsqv / n) - (mean * mean)
            return float(math.sqrt(max(var, 0.0)))

        rows.append({
            "date": pd.to_datetime(day),
            "total_rows": a["total_rows"],
            "active_rows": a["active_rows"],
            "active_ratio": (a["active_rows"] / a["total_rows"]) if a["total_rows"] else 0.0,

            "upper_error_std": calc_std(a["upper_err_sum"], a["upper_err_sumsq"], a["upper_err_n"]),
            "lower_error_std": calc_std(a["lower_err_sum"], a["lower_err_sumsq"], a["lower_err_n"]),
            "upper_vel_std": calc_std(a["upper_vel_sum"], a["upper_vel_sumsq"], a["upper_vel_n"]),
            "lower_vel_std": calc_std(a["lower_vel_sum"], a["lower_vel_sumsq"], a["lower_vel_n"]),

            "upper_error_spikes": a["upper_err_spikes"],
            "lower_error_spikes": a["lower_err_spikes"],
            "upper_current_spikes": a["upper_cur_spikes"],
            "lower_current_spikes": a["lower_cur_spikes"],
        })

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Gate: non-operational days should not be scored
    daily["is_operational_day"] = daily["active_rows"] >= MIN_ACTIVE_ROWS_PER_DAY

    def normalize_col(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        if x.dropna().empty:
            return x * 0
        lo = x.quantile(0.05)
        hi = x.quantile(0.95)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
            return x * 0
        return ((x - lo) / (hi - lo)).clip(0, 1)

    op = daily[daily["is_operational_day"]].copy()

    cols_for_score = [
        "upper_error_std", "lower_error_std", "upper_vel_std", "lower_vel_std",
        "upper_error_spikes", "lower_error_spikes",
        "upper_current_spikes", "lower_current_spikes"
    ]
    for c in cols_for_score:
        if c not in op.columns:
            op[c] = np.nan

    comps = {c: normalize_col(op[c]).reindex(op.index) for c in cols_for_score}

    op["oscillation_score"] = (
        0.35 * comps["upper_error_std"] +
        0.15 * comps["lower_error_std"] +
        0.15 * comps["upper_vel_std"] +
        0.10 * comps["lower_vel_std"] +
        0.10 * comps["upper_error_spikes"] +
        0.05 * comps["lower_error_spikes"] +
        0.05 * comps["upper_current_spikes"] +
        0.05 * comps["lower_current_spikes"]
    ).fillna(0.0)

    daily["oscillation_score"] = np.nan
    daily.loc[op.index, "oscillation_score"] = op["oscillation_score"].values

    daily.to_csv(OUT_DAILY, index=False)
    print(f"\nSaved: {OUT_DAILY}")

    return daily


def make_events(daily: pd.DataFrame) -> pd.DataFrame:
    op = daily[daily["is_operational_day"] & daily["oscillation_score"].notna()].copy()
    if op.empty:
        events = daily.iloc[0:0].copy()
        events.to_csv(OUT_EVENTS, index=False)
        print(f"Saved: {OUT_EVENTS} (empty)")
        return events

    thr = float(op["oscillation_score"].quantile(0.95))
    events = op[op["oscillation_score"] >= thr].sort_values("oscillation_score", ascending=False).copy()

    events.to_csv(OUT_EVENTS, index=False)
    print(f"Saved: {OUT_EVENTS}")

    print("\nTop oscillation (OPERATIONAL days only):\n")
    print(events.head(15)[[
        "date", "active_rows", "oscillation_score",
        "upper_error_std", "upper_error_spikes", "upper_current_spikes",
        "lower_error_std", "lower_error_spikes", "lower_current_spikes"
    ]].to_string(index=False))

    return events


def plot_trend(daily: pd.DataFrame):
    d = daily.copy()
    plt.figure(figsize=(13, 6))
    plt.plot(d["date"], d["oscillation_score"])
    plt.title("Oscillation Score Trend (operational days only; inactive=NaN)")
    plt.ylabel("oscillation_score (0-1)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_TREND)
    plt.close()
    print(f"Saved: {PLOT_TREND}")


def plot_window(daily: pd.DataFrame, start: str, end: str, outpath: Path, title: str):
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    w = d[(d["date"] >= pd.to_datetime(start)) & (d["date"] <= pd.to_datetime(end))].copy()

    plt.figure(figsize=(13, 6))
    plt.plot(w["date"], w["oscillation_score"])
    plt.title(title)
    plt.ylabel("oscillation_score (0-1)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def main():
    daily = compute_daily_oscillation()
    _events = make_events(daily)

    print("\nGenerating plots...")
    plot_trend(daily)

    plot_window(
        daily,
        start="2023-03-15",
        end="2023-06-15",
        outpath=PLOT_2023,
        title="Oscillation Score Window (Mar–Jun 2023)"
    )
    plot_window(
        daily,
        start="2026-02-15",
        end="2026-04-15",
        outpath=PLOT_2026,
        title="Oscillation Score Window (Feb–Apr 2026)"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()