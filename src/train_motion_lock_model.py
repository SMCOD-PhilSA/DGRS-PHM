# ============================================
# DGS Motion Lock Predictor
# Author: Arcee Juan
# ============================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

INPUT_FILE = "metrics_daily_features.csv"

MODEL_FILE = "motion_lock_model.pkl"


# ============================================
# KNOWN MOTION LOCK EVENTS
# ============================================

FAILURE_DATES = [
    "2022-11-25",
    "2023-04-05",
    "2025-06-06",
    "2026-03-04"
]

WINDOW_DAYS = 3


# ============================================
# CREATE FAILURE LABELS
# ============================================

def create_labels(df):

    df["date"] = pd.to_datetime(df["date"])

    df["motion_lock_label"] = 0

    for f in FAILURE_DATES:

        failure_date = pd.to_datetime(f)

        window_start = failure_date - pd.Timedelta(days=WINDOW_DAYS)

        mask = (
            (df["date"] >= window_start) &
            (df["date"] < failure_date)
        )

        df.loc[mask, "motion_lock_label"] = 1

    return df


# ============================================
# FEATURE SELECTION
# ============================================

def select_features(df):

    ignore_cols = [
        "date",
        "motion_lock_label"
    ]

    features = [c for c in df.columns if c not in ignore_cols]

    return features


# ============================================
# TRAIN MODEL
# ============================================

def train_model(df):

    features = select_features(df)

    X = df[features].fillna(0)

    y = df["motion_lock_label"]

    model = RandomForestClassifier(

        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    preds = model.predict(X)

    print("\n==============================")
    print("Classification Report")
    print("==============================\n")

    print(classification_report(y, preds))

    joblib.dump(model, MODEL_FILE)

    print("\nSaved model:", MODEL_FILE)

    return model, features


# ============================================
# FEATURE IMPORTANCE
# ============================================

def show_feature_importance(model, features):

    importance = pd.Series(
        model.feature_importances_,
        index=features
    )

    importance = importance.sort_values(ascending=False)

    print("\n==============================")
    print("Feature Importance")
    print("==============================\n")

    print(importance.head(20))


# ============================================
# MAIN
# ============================================

def main():

    print("\n==============================")
    print("Training Motion Lock Model")
    print("==============================\n")

    df = pd.read_csv(INPUT_FILE)

    df = create_labels(df)

    model, features = train_model(df)

    show_feature_importance(model, features)


if __name__ == "__main__":
    main()