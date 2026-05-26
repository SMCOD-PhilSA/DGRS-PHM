# ==========================================================
# DGS Predictive Health Monitoring
# Remaining Useful Life (RUL) Estimator
#
# Predicts number of days until next fault event
#
# Inputs (PROJECT ROOT)
#   daily_health_features.csv
#   degradation_state_history.csv
#
# Outputs
#   rul_training_dataset.csv
#   rul_predictions_full.csv
#   rul_predictions_latest.csv
#   rul_feature_importance.csv
#   rul_model_report.txt
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DAILY_FEATURES = BASE_DIR / "daily_health_features.csv"
DEGRADATION = BASE_DIR / "degradation_state_history.csv"

OUT_TRAIN = BASE_DIR / "rul_training_dataset.csv"
OUT_FULL = BASE_DIR / "rul_predictions_full.csv"
OUT_LATEST = BASE_DIR / "rul_predictions_latest.csv"
OUT_IMPORTANCE = BASE_DIR / "rul_feature_importance.csv"
OUT_REPORT = BASE_DIR / "rul_model_report.txt"


# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

FAULT_EVENT_THRESHOLD = 5
MAX_RUL_DAYS = 60
MIN_ROWS = 120

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)


# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def compute_time_to_next_event(df, event_col):
    """
    Computes days until next event day.
    """

    dates = df["date"].values
    events = np.where(df[event_col].values == 1)[0]

    next_event_index = np.full(len(df), -1)

    pointer = 0

    for i in range(len(df)):

        while pointer < len(events) and events[pointer] < i:
            pointer += 1

        if pointer < len(events):
            next_event_index[i] = events[pointer]

    rul = []

    for i, j in enumerate(next_event_index):

        if j == -1:
            rul.append(np.nan)
        else:
            d1 = pd.Timestamp(dates[i])
            d2 = pd.Timestamp(dates[j])
            rul.append((d2 - d1).days)

    return pd.Series(rul)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():

    print("\n==============================")
    print("RUL Estimator (Days to Next Fault Event)")
    print("==============================\n")

    print("Loading inputs...")

    daily = load_csv(DAILY_FEATURES).sort_values("date")
    deg = load_csv(DEGRADATION).sort_values("date")

    df = pd.merge(
        daily,
        deg[["date", "degradation_state", "daily_stress", "daily_damage", "daily_recovery"]],
        on="date",
        how="inner"
    )

    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < MIN_ROWS:
        raise RuntimeError("Not enough rows for training")

    # ------------------------------------------------------
    # DEFINE EVENT DAYS
    # ------------------------------------------------------

    df["fault_event_day"] = (
        df["total_episode_count"].fillna(0).astype(float) >= FAULT_EVENT_THRESHOLD
    ).astype(int)

    # ------------------------------------------------------
    # BUILD RUL LABEL
    # ------------------------------------------------------

    df["rul_days"] = compute_time_to_next_event(df, "fault_event_day")

    df["rul_days_capped"] = df["rul_days"].clip(0, MAX_RUL_DAYS)

    train_df = df.dropna(subset=["rul_days_capped"]).copy()

    train_df.to_csv(OUT_TRAIN, index=False)

    print("Saved training dataset:", OUT_TRAIN.name)

    # ------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------

    drop_cols = ["date", "fault_event_day", "rul_days", "rul_days_capped"]

    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X = train_df[feature_cols].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)

    # remove constant columns
    nunique = X.nunique()

    keep = nunique[nunique > 1].index

    X = X[keep]

    y = train_df["rul_days_capped"].astype(float).values

    # ------------------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------------------

    print("\nTraining model with time-series splits...")

    tscv = TimeSeriesSplit(n_splits=5)

    maes = []
    rmses = []

    for fold, (tr, te) in enumerate(tscv.split(X), 1):

        model = RandomForestRegressor(**RF_PARAMS)

        model.fit(X.iloc[tr], y[tr])

        pred = model.predict(X.iloc[te])

        mae = mean_absolute_error(y[te], pred)

        rmse = np.sqrt(mean_squared_error(y[te], pred))

        maes.append(mae)
        rmses.append(rmse)

        print(f"Fold {fold} | MAE={mae:.2f} days | RMSE={rmse:.2f} days")

    # ------------------------------------------------------
    # FINAL MODEL
    # ------------------------------------------------------

    model = RandomForestRegressor(**RF_PARAMS)

    model.fit(X, y)

    # ------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values("importance", ascending=False)

    importance.to_csv(OUT_IMPORTANCE, index=False)

    # ------------------------------------------------------
    # PREDICT FOR ALL DAYS
    # ------------------------------------------------------

    X_all = df[X.columns].copy()

    for col in X_all.columns:
        X_all[col] = pd.to_numeric(X_all[col], errors="coerce")

    X_all = X_all.fillna(0)

    df["rul_prediction_days"] = model.predict(X_all).clip(0, MAX_RUL_DAYS)

    # ------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------

    out = df[[
        "date",
        "fault_event_day",
        "rul_days",
        "rul_prediction_days",
        "degradation_state",
        "total_episode_count",
        "motion_event_count",
        "track_event_count",
        "rolling_7d_episode_count"
    ]]

    out.to_csv(OUT_FULL, index=False)

    latest = out.tail(1)

    latest.to_csv(OUT_LATEST, index=False)

    # ------------------------------------------------------
    # REPORT
    # ------------------------------------------------------

    report = []

    report.append("RUL MODEL REPORT")
    report.append(f"Generated: {datetime.utcnow()}")
    report.append("")
    report.append(f"MAE mean: {np.mean(maes):.2f}")
    report.append(f"RMSE mean: {np.mean(rmses):.2f}")
    report.append("")
    report.append("Top Predictors:")

    for _, r in importance.head(15).iterrows():
        report.append(f"{r.feature} : {r.importance:.4f}")

    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(report))

    print("\nSaved outputs:")
    print("  ", OUT_FULL.name)
    print("  ", OUT_LATEST.name)
    print("  ", OUT_IMPORTANCE.name)
    print("  ", OUT_REPORT.name)

    print("\nLatest prediction:\n")
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()