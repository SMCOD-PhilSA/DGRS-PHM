# ============================================
# DGS Predictive Health Monitoring
# 7-Day Degrading State ML Trainer
# Author: Arcee Juan
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# ==============================
# CONFIG
# ==============================

FEATURE_FILE = Path("daily_health_features.csv")
MODEL_OUTPUT = Path("ml_model_7day.joblib")
HORIZON_DAYS = 7
N_SPLITS = 5

# Threshold to define degrading state
DEGRADING_THRESHOLD = 5  # rolling 7-day episodes


# ==============================
# Build 7-Day Target
# ==============================

def build_target(df):

    df = df.sort_values("date").reset_index(drop=True)

    # Define degrading state directly from episode activity
    df["is_degrading"] = (
        df["rolling_7d_episode_count"] >= DEGRADING_THRESHOLD
    ).astype(int)

    future_target = []

    for i in range(len(df)):
        future_window = df["is_degrading"].iloc[i+1:i+1+HORIZON_DAYS]
        future_target.append(1 if future_window.sum() > 0 else 0)

    df["target_7d"] = future_target

    return df


# ==============================
# Main Training
# ==============================

def main():

    print("Loading daily health features...")

    df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])

    df = build_target(df)

    # Remove last horizon rows
    df = df.iloc[:-HORIZON_DAYS]

    feature_cols = [
        "total_episode_count",
        "rolling_7d_episode_count",
        "total_fault_duration_sec",
        "episode_trend_slope_14d",
        "unique_error_codes"
    ]

    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df["target_7d"]

    print("\nTraining samples:", len(X))
    print("Positive (degrading within 7d):", y.sum())
    print("Negative:", len(y) - y.sum())

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    aucs = []
    aps = []

    print("\nTime-Series Cross Validation:\n")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() < 2:
            print(f"Skipping fold {fold}: only one class in training")
            continue

        pipeline.fit(X.iloc[train_idx], y_train)

        proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]

        if y_test.nunique() < 2:
            print(f"Skipping fold {fold}: only one class in test")
            continue

        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)

        aucs.append(auc)
        aps.append(ap)

        print(f"Fold {fold} - ROC-AUC: {auc:.4f} | Avg Precision: {ap:.4f}")

    if len(aucs) > 0:
        print("\n==============================")
        print("Mean ROC-AUC:", round(np.mean(aucs), 4))
        print("Mean Avg Precision:", round(np.mean(aps), 4))
        print("==============================")
    else:
        print("\nNo valid folds available.")

    print("\nTraining final model on full dataset...")
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_OUTPUT)
    print("Model saved to:", MODEL_OUTPUT.resolve())


if __name__ == "__main__":
    main()