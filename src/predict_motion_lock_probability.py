# ============================================
# DGS Motion Lock Predictor
# Author: Arcee Juan
# ============================================

import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

MODEL_FILE = "motion_lock_model.pkl"
FEATURE_FILE = "metrics_daily_features.csv"


# ============================================
# LOAD MODEL
# ============================================

def load_model():

    model = joblib.load(MODEL_FILE)

    print("\nLoaded model:", MODEL_FILE)

    return model


# ============================================
# LOAD FEATURES
# ============================================

def load_features():

    df = pd.read_csv(FEATURE_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    return df


# ============================================
# SELECT MODEL FEATURES
# ============================================

def get_feature_columns(df):

    ignore = [
        "date",
        "motion_lock_label"
    ]

    features = [c for c in df.columns if c not in ignore]

    return features


# ============================================
# COMPUTE RISK
# ============================================

def compute_risk(model, df):

    features = get_feature_columns(df)

    latest = df.iloc[-1:]

    X = latest[features].fillna(0)

    prob = model.predict_proba(X)[0][1]

    return prob, latest


# ============================================
# INTERPRET RISK LEVEL
# ============================================

def risk_level(prob):

    if prob < 0.2:
        return "LOW"

    elif prob < 0.5:
        return "MODERATE"

    elif prob < 0.75:
        return "HIGH"

    else:
        return "CRITICAL"


# ============================================
# DISPLAY PHM STATUS
# ============================================

def show_status(prob, latest):

    date = latest["date"].values[0]

    failed_moves = latest.get("failed_move_count", pd.Series([0])).values[0]

    tracking_ratio = latest.get("tracking_ratio", pd.Series([0])).values[0]

    error_spikes = latest.get("error_spikes", pd.Series([0])).values[0]

    print("\n=======================================")
    print("DGS ANTENNA PHM STATUS")
    print("=======================================\n")

    print("Date:", date)

    print("\nKey Signals")

    print("Tracking ratio:", round(tracking_ratio,3))
    print("Failed movement attempts:", int(failed_moves))
    print("Error spikes:", int(error_spikes))

    print("\nMotion Lock Risk")

    print("Probability:", round(prob,3))

    print("Risk Level:", risk_level(prob))


# ============================================
# FORECAST 72H RISK
# ============================================

def forecast_risk(prob):

    print("\nRisk Forecast")

    risk24 = prob * 0.6
    risk48 = prob * 0.85
    risk72 = prob

    print("24h risk:", round(risk24,3))
    print("48h risk:", round(risk48,3))
    print("72h risk:", round(risk72,3))


# ============================================
# MAIN
# ============================================

def main():

    print("\n==============================")
    print("DGS Motion Lock Prediction")
    print("==============================")

    model = load_model()

    df = load_features()

    prob, latest = compute_risk(model, df)

    show_status(prob, latest)

    forecast_risk(prob)


if __name__ == "__main__":
    main()