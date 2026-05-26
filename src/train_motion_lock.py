# ============================================
# Motion Lock ML Model Trainer (Error-Code Based)
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = BASE_DIR / "metrics_daily_features.csv"
STRESS_FILE = BASE_DIR / "metrics_servo_stress_daily.csv"
FAULT_FILE = BASE_DIR / "fault_episodes.csv"

MODEL_OUT = BASE_DIR / "motion_lock_model.pkl"


# ============================================
# LOAD DATA
# ============================================

def load_data():

    features = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    stress = pd.read_csv(STRESS_FILE, parse_dates=["date"])

    df = pd.merge(features, stress, on="date", how="left")

    faults = pd.read_csv(FAULT_FILE)

    if "start_time" in faults.columns:
        faults["date"] = pd.to_datetime(faults["start_time"]).dt.date
    elif "date" in faults.columns:
        faults["date"] = pd.to_datetime(faults["date"]).dt.date

    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df, faults


# ============================================
# EXTRA FEATURES
# ============================================

def compute_extra_features(df):

    stress = df["servo_stress_index"]

    df["stress_slope"] = stress.diff()
    df["stress_volatility"] = stress.rolling(5).std()

    df = df.fillna(0)

    return df


# ============================================
# BUILD FAILURE LABELS
# ============================================

def build_labels(df, faults):

    # Servo fault codes associated with motion lock
    servo_fault_codes = [7490, 7901, 7452, 7412]

    faults = faults.copy()

    faults["error_code"] = pd.to_numeric(
        faults["error_code"], errors="coerce"
    )

    motion_faults = faults[
        faults["error_code"].isin(servo_fault_codes)
    ]

    motion_days = set(pd.to_datetime(motion_faults["date"]).dt.date)

    df["motion_lock_next3d"] = 0

    for i, row in df.iterrows():

        for d in range(1,4):

            future_day = row["date"] + pd.Timedelta(days=d)

            if future_day in motion_days:

                df.loc[i,"motion_lock_next3d"] = 1
                break

    return df


# ============================================
# SELECT FEATURES
# ============================================

def select_features(df):

    candidate = [

        "servo_stress_index",
        "stress_slope",
        "stress_volatility",

        "Upper/X following error_mean",
        "Upper/X following error_max",

        "Lower/Y following error_mean",
        "Lower/Y following error_max",

        "Upper axis current_mean",
        "Lower axis current_mean",

        "Upper/X velocity_mean",
        "Lower/Y velocity_mean",

        "Cabinet temperature_mean"
    ]

    features = [c for c in candidate if c in df.columns]

    return features


# ============================================
# TRAIN MODEL
# ============================================

def train_model(df, features):

    X = df[features]
    y = df["motion_lock_next3d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nClassification Report\n")
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_OUT)

    print("\nSaved model:", MODEL_OUT)

    print("\nFeature Importance\n")

    for f, imp in sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{f:35} {imp:.3f}")


# ============================================
# MAIN
# ============================================

def main():

    print("\n==============================")
    print("Training Motion Lock Model")
    print("==============================")

    df, faults = load_data()

    df = compute_extra_features(df)

    df = build_labels(df, faults)

    features = select_features(df)

    print("\nUsing features:")
    for f in features:
        print("-", f)

    train_model(df, features)


if __name__ == "__main__":
    main()