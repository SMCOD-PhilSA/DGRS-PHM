# ============================================
# DGS Servo Oscillation Detector (FIXED)
# Author: Arcee Juan
# ============================================

import pandas as pd
import numpy as np

INPUT_FILE = "metrics_master.csv"
OUTPUT_FILE = "servo_oscillation_daily.csv"

CHUNK_SIZE = 200000


# ============================================
# SAFE NUMERIC
# ============================================

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ============================================
# DERIVATIVE-BASED OSCILLATION
# ============================================

def derivative_spike_count(series, z_thresh=3.0):
    """
    Count rapid changes using first derivative.
    A spike means the derivative is unusually large
    relative to its own distribution.
    """
    s = to_numeric(series).dropna()

    if len(s) < 5:
        return 0, 0.0

    d = s.diff().dropna()

    if len(d) < 5:
        return 0, 0.0

    mu = d.mean()
    sigma = d.std()

    if pd.isna(sigma) or sigma == 0:
        return 0, float(d.std())

    spikes = (np.abs(d - mu) > z_thresh * sigma).sum()

    return int(spikes), float(d.std())


def sign_flip_rate(series):
    """
    Count how often the derivative changes sign.
    This is a simple proxy for hunting / oscillation.
    """
    s = to_numeric(series).dropna()

    if len(s) < 5:
        return 0

    d = s.diff().dropna()

    if len(d) < 5:
        return 0

    sign = np.sign(d)

    flips = (sign.shift(1) * sign < 0).sum()

    return int(flips)


# ============================================
# MAIN DETECTOR
# ============================================

def detect_oscillation():

    print("\nReading telemetry...")

    reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    rows = []

    for chunk in reader:

        chunk["Time"] = pd.to_datetime(chunk["Time"], errors="coerce")
        chunk = chunk.dropna(subset=["Time"])

        if chunk.empty:
            continue

        chunk["date"] = chunk["Time"].dt.date

        numeric_cols = [
            "Upper axis current",
            "Lower axis current",
            "Upper/X velocity",
            "Lower/Y velocity",
            "Upper/X following error",
            "Lower/Y following error"
        ]

        for c in numeric_cols:
            if c in chunk.columns:
                chunk[c] = to_numeric(chunk[c])

        grouped = chunk.groupby("date")

        for date, g in grouped:

            row = {}
            row["date"] = date

            # ------------------------------------
            # Velocity oscillation
            # ------------------------------------
            upper_vel_spikes, upper_vel_std = derivative_spike_count(
                g["Upper/X velocity"]
            ) if "Upper/X velocity" in g.columns else (0, 0.0)

            lower_vel_spikes, lower_vel_std = derivative_spike_count(
                g["Lower/Y velocity"]
            ) if "Lower/Y velocity" in g.columns else (0, 0.0)

            upper_vel_flips = sign_flip_rate(
                g["Upper/X velocity"]
            ) if "Upper/X velocity" in g.columns else 0

            lower_vel_flips = sign_flip_rate(
                g["Lower/Y velocity"]
            ) if "Lower/Y velocity" in g.columns else 0

            # ------------------------------------
            # Following error oscillation
            # ------------------------------------
            upper_err_spikes, upper_err_std = derivative_spike_count(
                g["Upper/X following error"]
            ) if "Upper/X following error" in g.columns else (0, 0.0)

            lower_err_spikes, lower_err_std = derivative_spike_count(
                g["Lower/Y following error"]
            ) if "Lower/Y following error" in g.columns else (0, 0.0)

            upper_err_flips = sign_flip_rate(
                g["Upper/X following error"]
            ) if "Upper/X following error" in g.columns else 0

            lower_err_flips = sign_flip_rate(
                g["Lower/Y following error"]
            ) if "Lower/Y following error" in g.columns else 0

            # ------------------------------------
            # Current oscillation
            # ------------------------------------
            upper_cur_spikes, upper_cur_std = derivative_spike_count(
                g["Upper axis current"]
            ) if "Upper axis current" in g.columns else (0, 0.0)

            lower_cur_spikes, lower_cur_std = derivative_spike_count(
                g["Lower axis current"]
            ) if "Lower axis current" in g.columns else (0, 0.0)

            upper_cur_flips = sign_flip_rate(
                g["Upper axis current"]
            ) if "Upper axis current" in g.columns else 0

            lower_cur_flips = sign_flip_rate(
                g["Lower axis current"]
            ) if "Lower axis current" in g.columns else 0

            # ------------------------------------
            # Save raw daily indicators
            # ------------------------------------
            row["upper_vel_spikes"] = upper_vel_spikes
            row["lower_vel_spikes"] = lower_vel_spikes
            row["upper_vel_flip_rate"] = upper_vel_flips
            row["lower_vel_flip_rate"] = lower_vel_flips
            row["upper_vel_deriv_std"] = upper_vel_std
            row["lower_vel_deriv_std"] = lower_vel_std

            row["upper_err_spikes"] = upper_err_spikes
            row["lower_err_spikes"] = lower_err_spikes
            row["upper_err_flip_rate"] = upper_err_flips
            row["lower_err_flip_rate"] = lower_err_flips
            row["upper_err_deriv_std"] = upper_err_std
            row["lower_err_deriv_std"] = lower_err_std

            row["upper_cur_spikes"] = upper_cur_spikes
            row["lower_cur_spikes"] = lower_cur_spikes
            row["upper_cur_flip_rate"] = upper_cur_flips
            row["lower_cur_flip_rate"] = lower_cur_flips
            row["upper_cur_deriv_std"] = upper_cur_std
            row["lower_cur_deriv_std"] = lower_cur_std

            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("No oscillation data computed.")
        df.to_csv(OUTPUT_FILE, index=False)
        return

    df = df.sort_values("date")

    # ----------------------------------------
    # Normalize each component
    # ----------------------------------------
    score_cols = [
        "upper_vel_spikes", "lower_vel_spikes",
        "upper_vel_flip_rate", "lower_vel_flip_rate",
        "upper_err_spikes", "lower_err_spikes",
        "upper_err_flip_rate", "lower_err_flip_rate",
        "upper_cur_spikes", "lower_cur_spikes",
        "upper_cur_flip_rate", "lower_cur_flip_rate",
        "upper_vel_deriv_std", "lower_vel_deriv_std",
        "upper_err_deriv_std", "lower_err_deriv_std",
        "upper_cur_deriv_std", "lower_cur_deriv_std"
    ]

    for c in score_cols:
        if c not in df.columns:
            df[c] = 0.0

        lo = df[c].quantile(0.05)
        hi = df[c].quantile(0.95)

        if pd.isna(lo) or pd.isna(hi) or hi - lo == 0:
            df[c + "_norm"] = 0.0
        else:
            df[c + "_norm"] = ((df[c] - lo) / (hi - lo)).clip(0, 1)

    # ----------------------------------------
    # Build oscillation score
    # Emphasize following error + velocity,
    # then current
    # ----------------------------------------
    df["oscillation_score"] = (
        0.25 * df["upper_err_flip_rate_norm"] +
        0.15 * df["lower_err_flip_rate_norm"] +
        0.20 * df["upper_vel_flip_rate_norm"] +
        0.10 * df["lower_vel_flip_rate_norm"] +
        0.10 * df["upper_cur_flip_rate_norm"] +
        0.05 * df["lower_cur_flip_rate_norm"] +
        0.10 * df["upper_err_deriv_std_norm"] +
        0.05 * df["upper_vel_deriv_std_norm"]
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)

    print("\nTop oscillation days:\n")
    print(
        df.sort_values("oscillation_score", ascending=False)
        .head(15)[[
            "date",
            "upper_err_flip_rate",
            "upper_vel_flip_rate",
            "upper_cur_flip_rate",
            "oscillation_score"
        ]]
    )


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    detect_oscillation()