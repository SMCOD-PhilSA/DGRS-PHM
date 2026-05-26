import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor


BASE = Path(__file__).resolve().parent.parent

# input files
DAILY_HEALTH = BASE / "daily_health_features.csv"
DEGRADATION = BASE / "degradation_state_history.csv"
METRICS = BASE / "metrics_servo_stress_daily.csv"

# output
OUT = BASE / "rul_predictions_fused.csv"


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

def load_data():

    print("Loading datasets...")

    health = pd.read_csv(DAILY_HEALTH, parse_dates=["date"])
    deg = pd.read_csv(DEGRADATION, parse_dates=["date"])
    metrics = pd.read_csv(METRICS, parse_dates=["date"])

    df = health.merge(deg, on="date", how="inner")

    df = df.merge(metrics, on="date", how="left")

    df = df.sort_values("date")

    return df


# ------------------------------------------------
# BUILD EVENT LABEL
# ------------------------------------------------

def build_fault_label(df):

    df["fault_event"] = (df["total_episode_count"] >= 5).astype(int)

    return df


# ------------------------------------------------
# BUILD RUL LABEL
# ------------------------------------------------

def compute_rul(df):

    event_indices = df.index[df["fault_event"] == 1].tolist()

    rul = []

    for i in range(len(df)):

        future = [e for e in event_indices if e >= i]

        if len(future) == 0:
            rul.append(np.nan)

        else:
            rul.append(future[0] - i)

    df["rul_days"] = rul

    return df


# ------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------

def train_model(df):

    train = df.dropna(subset=["rul_days"])

    drop = ["date", "fault_event", "rul_days"]

    features = [c for c in train.columns if c not in drop]

    X = train[features].fillna(0)

    y = train["rul_days"]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model, features


# ------------------------------------------------
# PREDICT RUL
# ------------------------------------------------

def predict(df, model, features):

    X = df[features].fillna(0)

    df["rul_prediction"] = model.predict(X)

    return df


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    df = load_data()

    df = build_fault_label(df)

    df = compute_rul(df)

    model, features = train_model(df)

    df = predict(df, model, features)

    df.to_csv(OUT, index=False)

    print("\nSaved:", OUT.name)

    latest = df.tail(1)

    print("\nLatest Prediction\n")

    print(latest[[
        "date",
        "rul_prediction",
        "degradation_state",
        "total_episode_count"
    ]])


if __name__ == "__main__":
    main()