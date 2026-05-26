# ============================================
# DGS Predictive Health Monitoring
# Degradation Rate Model (α estimation)
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FEATURE_FILE = Path("daily_health_features.csv")
OUTPUT_PARAMS = Path("degradation_params.csv")


def load_data():
    df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric
    numeric_cols = [
        "rolling_7d_episode_count",
        "motion_event_count",
        "track_event_count",
        "total_fault_duration_sec",
        "active_flag"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert active_flag to numeric (True=1, False=0)
    df["active_flag"] = df["active_flag"].astype(float)

    # Compute degradation rate
    df["delta_degradation"] = df["rolling_7d_episode_count"].diff()

    # Drop rows with ANY NaN in relevant columns
    df = df.dropna(subset=numeric_cols + ["delta_degradation"])

    return df


def train_model(df):

    feature_cols = [
        "motion_event_count",
        "track_event_count",
        "total_fault_duration_sec",
        "active_flag"
    ]

    X = df[feature_cols]
    y = df["delta_degradation"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    return model, feature_cols, r2


def save_params(model, feature_cols, r2):

    params = dict(zip(feature_cols, model.coef_))
    params["intercept"] = model.intercept_
    params["r2"] = r2

    pd.DataFrame(
        params.items(),
        columns=["parameter", "value"]
    ).to_csv(OUTPUT_PARAMS, index=False)


def main():

    print("\n===================================")
    print("Degradation Rate Model Training")
    print("===================================")

    df = load_data()

    print("Training samples:", len(df))

    model, feature_cols, r2 = train_model(df)

    print("\nLearned Coefficients:")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"{name}: {coef:.6f}")

    print(f"\nIntercept: {model.intercept_:.6f}")
    print(f"Model R²: {r2:.4f}")

    save_params(model, feature_cols, r2)

    print("\nSaved parameters to:", OUTPUT_PARAMS.resolve())


if __name__ == "__main__":
    main()