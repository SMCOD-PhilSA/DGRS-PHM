# ============================================
# DGS PHM - Bounded Physics-Based Degradation Model
# Author: Arcee Juan
# ============================================

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DAILY_FEATURES = BASE_DIR / "daily_health_features.csv"
REST_EVENTS = BASE_DIR / "rest_recovery_events.csv"

OUT_STATE_HISTORY = BASE_DIR / "degradation_state_history.csv"
OUT_PARAMS = BASE_DIR / "physics_params.csv"


# --------------------------------------------
# Safe CSV Loader
# --------------------------------------------

def safe_read_csv(path: Path, parse_dates=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates or [])


# --------------------------------------------
# Smooth Normalization (NO hard clipping saturation)
# --------------------------------------------

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo = np.nanpercentile(s, 5)
    hi = np.nanpercentile(s, 95)

    if hi - lo <= 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)

    x = (s - lo) / (hi - lo)

    # Smooth compression instead of clip
    return np.tanh(1.5 * x).clip(0, 1)


# --------------------------------------------
# Recovery Estimation (optional)
# --------------------------------------------

def estimate_rest_recovery(rest_df: pd.DataFrame) -> dict:
    cols = set(rest_df.columns)

    if "next_health_index" in cols:
        rec = rest_df["next_health_index"].astype(float)
        rec_mean = float(np.nanmean(rec))
        rec_std = float(np.nanstd(rec))
        return {
            "recovery_signal": "health_index",
            "recovery_mean": rec_mean,
            "recovery_std": rec_std,
        }

    return {
        "recovery_signal": "none",
        "recovery_mean": 0.0,
        "recovery_std": 0.0,
    }


# --------------------------------------------
# BOUNDED PHYSICS DEGRADATION MODEL
# --------------------------------------------

def build_degradation_state(df: pd.DataFrame, params: dict) -> pd.DataFrame:

    df = df.copy().sort_values("date")

    required_cols = [
        "total_episode_count",
        "total_fault_duration_sec",
        "unique_error_codes",
        "motion_event_count",
        "track_event_count",
        "rolling_7d_episode_count",
        "active_flag",
    ]

    for c in required_cols:
        if c not in df.columns:
            df[c] = 0

    # Normalize stress channels
    n_episode = normalize_series(df["total_episode_count"])
    n_dur = normalize_series(df["total_fault_duration_sec"])
    n_codes = normalize_series(df["unique_error_codes"])
    n_motion = normalize_series(df["motion_event_count"])
    n_track = normalize_series(df["track_event_count"])
    n_roll = normalize_series(df["rolling_7d_episode_count"])

    # Stress weights (stable proportions)
    stress = (
        0.35 * n_episode +
        0.20 * n_dur +
        0.15 * n_codes +
        0.10 * n_motion +
        0.10 * n_track +
        0.10 * n_roll
    ).clip(0, 1)

    damage_scale = float(params.get("damage_scale", 1.5))
    recovery_scale = float(params.get("recovery_scale", 4.5))

    daily_damage = stress * damage_scale
    is_active = df["active_flag"].astype(bool).values
    daily_recovery = np.where(is_active, 0.0, recovery_scale)

    # ----------------------------------------
    # Bounded degradation dynamics
    # ----------------------------------------
    D = []
    d = 10.0  # Start at low degradation instead of 0 or 100

    for dmg, rec in zip(daily_damage.values, daily_recovery):

        # Asymptotic growth & decay
        d = d + dmg * (1 - d/100.0) - rec * (d/100.0)

        d = max(0.0, min(100.0, d))
        D.append(d)

    out = df[["date"]].copy()
    out["degradation_state"] = np.round(D, 3)
    out["daily_stress"] = np.round(stress.values, 6)
    out["daily_damage"] = np.round(daily_damage.values, 6)
    out["daily_recovery"] = np.round(daily_recovery, 6)

    return out


# --------------------------------------------
# MAIN
# --------------------------------------------

def main():
    print("\n===================================")
    print("Bounded Physics Degradation Model")
    print("===================================")

    df = safe_read_csv(DAILY_FEATURES, parse_dates=["date"])

    if REST_EVENTS.exists():
        rest_df = safe_read_csv(REST_EVENTS)
        recovery_stats = estimate_rest_recovery(rest_df)
    else:
        recovery_stats = {"recovery_signal": "none"}

    params = {
        "damage_scale": 1.5,   # Reduced from 4.0
        "recovery_scale": 4.5
    }

    state_hist = build_degradation_state(df, params)

    state_hist.to_csv(OUT_STATE_HISTORY, index=False)

    param_rows = []
    for k, v in params.items():
        param_rows.append({"parameter": k, "value": v})
    pd.DataFrame(param_rows).to_csv(OUT_PARAMS, index=False)

    print("Saved:", OUT_STATE_HISTORY.resolve())
    print("Saved:", OUT_PARAMS.resolve())

    print("\nLatest state:")
    print(state_hist.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()