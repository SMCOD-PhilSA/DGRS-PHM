# Author: Arcee Juan
# Weekly Risk Forecast (7-day Critical probability) - Empirical calibration

import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "daily_health_with_index.csv"
OUTPUT_FILE = BASE_DIR / "weekly_risk_forecast.csv"

HORIZON_DAYS = 7
N_BINS = 10  # deciles


def compute_future_critical_label(df: pd.DataFrame) -> pd.Series:
    """
    Label each day as 1 if within the next HORIZON_DAYS there exists at least
    one ACTIVE day that is Critical.
    """
    is_critical_active = (df["risk_level"] == "Critical") & (df["active_flag"].astype(bool))

    # forward-looking window: for each i, check any critical in (i+1 ... i+H)
    y = np.zeros(len(df), dtype=int)
    crit = is_critical_active.to_numpy()

    for i in range(len(df)):
        j_end = min(len(df), i + 1 + HORIZON_DAYS)
        if i + 1 < j_end and crit[i + 1 : j_end].any():
            y[i] = 1
    return pd.Series(y, index=df.index, name=f"critical_next_{HORIZON_DAYS}d")


def build_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Risk score in [0, 1], higher = worse.
    Uses the already-computed health_index (smoothed, window-based).
    """
    # Health index is 0..100; convert to risk 0..1
    risk = 1 - (df["health_index"].clip(0, 100) / 100.0)
    return risk.clip(0, 1)


def calibrate_probability(risk_score: pd.Series, y: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """
    Empirical calibration: bin risk_score into quantiles and compute observed
    frequency of future critical events in each bin.
    """
    # Use only rows where y is defined (exclude last horizon days where label is less meaningful)
    valid_mask = y.notna()
    rs = risk_score[valid_mask].copy()
    yy = y[valid_mask].copy()

    # If risk scores are all identical, fallback to global rate
    if rs.nunique() < 2:
        base_p = float(yy.mean()) if len(yy) else 0.0
        return pd.DataFrame([{
            "bin_lo": 0.0,
            "bin_hi": 1.0,
            "p_critical_next_7d": base_p,
            "count": int(len(yy))
        }])

    # Quantile bins
    bins = pd.qcut(rs, q=n_bins, duplicates="drop")
    calib = (
        pd.DataFrame({"bin": bins, "y": yy, "risk_score": rs})
        .groupby("bin", observed=False)
        .agg(
            p_critical_next_7d=("y", "mean"),
            count=("y", "size"),
            bin_lo=("risk_score", "min"),
            bin_hi=("risk_score", "max"),
        )
        .reset_index(drop=True)
        .sort_values("bin_lo")
        .reset_index(drop=True)
    )

    # Light smoothing to avoid jagged probabilities (monotonic-ish)
    calib["p_critical_next_7d"] = calib["p_critical_next_7d"].rolling(2, min_periods=1).mean()

    return calib


def predict_prob(calib: pd.DataFrame, risk_value: float) -> float:
    """
    Map a risk_value to calibrated probability using nearest bin.
    """
    if calib.empty:
        return 0.0

    # Find bin where risk_value falls, else clamp to nearest
    in_bin = calib[(calib["bin_lo"] <= risk_value) & (risk_value <= calib["bin_hi"])]
    if not in_bin.empty:
        return float(in_bin.iloc[0]["p_critical_next_7d"])

    # Clamp
    if risk_value < float(calib["bin_lo"].min()):
        return float(calib.iloc[0]["p_critical_next_7d"])
    return float(calib.iloc[-1]["p_critical_next_7d"])


def classify_weekly(prob: float) -> str:
    """
    Weekly risk label based on probability of at least one Critical day in next 7 days.
    Tune thresholds later based on operations preference.
    """
    if prob >= 0.70:
        return "Critical"
    if prob >= 0.40:
        return "Warning"
    if prob >= 0.20:
        return "Watch"
    return "Normal"


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure types
    df["active_flag"] = df["active_flag"].astype(bool)

    # Build target and risk score
    y = compute_future_critical_label(df)
    df[y.name] = y

    df["risk_score"] = build_risk_score(df)

    # Calibrate on historical data where we have future horizon
    # exclude last HORIZON_DAYS since their label looks ahead beyond dataset end
    calib_df = df.iloc[:-HORIZON_DAYS].copy() if len(df) > HORIZON_DAYS else df.copy()
    calib = calibrate_probability(calib_df["risk_score"], calib_df[y.name], n_bins=N_BINS)

    # Predict probability for every day
    probs = []
    weekly_levels = []
    for rv in df["risk_score"].to_numpy():
        p = predict_prob(calib, float(rv))
        probs.append(p)
        weekly_levels.append(classify_weekly(p))

    df["p_critical_next_7d"] = probs
    df["weekly_forecast_level"] = weekly_levels

    # Save output
    out_cols = [
        "date",
        "active_flag",
        "health_index",
        "risk_level",
        "risk_score",
        "p_critical_next_7d",
        "weekly_forecast_level",
        "rolling_7d_episode_count",
        "rolling_7d_fault_duration",
        "episode_trend_slope_14d",
        y.name,
    ]
    df[out_cols].to_csv(OUTPUT_FILE, index=False)

    # Print latest
    latest = df.iloc[-1]
    print("Weekly forecast (next 7 days):")
    print("Date:", latest["date"])
    print("Active today:", bool(latest["active_flag"]))
    print("Health index:", round(float(latest["health_index"]), 2))
    print("Today risk level:", latest["risk_level"])
    print("P(Critical at least once in next 7 days):", round(float(latest["p_critical_next_7d"]), 3))
    print("Weekly forecast level:", latest["weekly_forecast_level"])
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()