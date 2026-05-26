# ============================================
# DGS Predictive Health Monitoring
# FINAL Health Projection Model (7-Day)
# ============================================

import pandas as pd
from pathlib import Path

# --------------------------------------------
# PATH CONFIG (matches your actual structure)
# --------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = BASE_DIR / "daily_health_features.csv"
LOAD_FILE = BASE_DIR / "daily_load_forecast.csv"
PARAM_FILE = BASE_DIR / "degradation_params.csv"
REST_PARAM_FILE = BASE_DIR / "rest_recovery_params.csv"

DEGRADING_THRESHOLD = 5
CRITICAL_THRESHOLD = 15


# --------------------------------------------
# Load Data
# --------------------------------------------

def load_latest_health():
    df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    df = df.sort_values("date")
    return df.iloc[-1]


def load_degradation_params():
    df = pd.read_csv(PARAM_FILE)

    if "parameter" in df.columns and "value" in df.columns:
        return dict(zip(df["parameter"], df["value"]))

    # fallback if saved as wide format
    if len(df) == 1:
        return df.iloc[0].to_dict()

    return {}


def load_forecast():
    df = pd.read_csv(LOAD_FILE, parse_dates=["date"])
    return df.sort_values("date")


# --------------------------------------------
# State Classification
# --------------------------------------------

def classify_state(value):
    if value >= CRITICAL_THRESHOLD:
        return "CRITICAL"
    elif value >= DEGRADING_THRESHOLD:
        return "DEGRADING"
    else:
        return "STABLE"


# --------------------------------------------
# Projection Engine
# --------------------------------------------

def project_health():

    latest = load_latest_health()
    params = load_degradation_params()
    forecast = load_forecast()

    rolling = latest["rolling_7d_episode_count"]

    results = []

    for _, row in forecast.iterrows():

        load_factor = row["load_score_day"]

        # Learned degradation equation
        delta = (
            params.get("motion_event_count", 0) * load_factor * 10 +
            params.get("track_event_count", 0) * load_factor * 10 +
            params.get("total_fault_duration_sec", 0) * load_factor * 100 +
            params.get("active_flag", 0) +
            params.get("intercept", 0)
        )

        rolling += delta

        if rolling < 0:
            rolling = 0

        state = classify_state(rolling)

        results.append({
            "date": row["date"],
            "projected_rolling_7d_episode_count": round(rolling, 3),
            "state": state
        })

    return latest, pd.DataFrame(results)


# --------------------------------------------
# Main
# --------------------------------------------

def main():

    print("\n===================================")
    print("DGS 7-Day Health Projection (FINAL)")
    print("===================================")

    latest, projection = project_health()

    print("Current Date:", latest["date"].date())
    print("Current rolling_7d_episode_count:",
          round(latest["rolling_7d_episode_count"], 3))

    print("\nProjected 7-Day State:\n")

    for _, row in projection.iterrows():
        print(f"{row['date'].date()} → {row['state']} "
              f"(rolling={row['projected_rolling_7d_episode_count']})")

    output = BASE_DIR / "health_projection_7day.csv"
    projection.to_csv(output, index=False)

    print("\nSaved projection to:", output.resolve())


if __name__ == "__main__":
    main()