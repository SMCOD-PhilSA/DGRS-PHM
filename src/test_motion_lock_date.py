# ============================================
# Test Motion Lock Prediction for Any Date
# ============================================

import pandas as pd
import joblib
import sys

MODEL_FILE = "motion_lock_model.pkl"
FEATURE_FILE = "metrics_daily_features.csv"


def load_data():

    df = pd.read_csv(FEATURE_FILE)

    df["date"] = pd.to_datetime(df["date"])

    return df


def load_model():

    return joblib.load(MODEL_FILE)


def get_features(df):

    ignore = ["date", "motion_lock_label"]

    return [c for c in df.columns if c not in ignore]


def test_date(test_date):

    df = load_data()

    model = load_model()

    features = get_features(df)

    row = df[df["date"] == pd.to_datetime(test_date)]

    if row.empty:

        print("Date not found in dataset")
        return

    X = row[features].fillna(0)

    prob = model.predict_proba(X)[0][1]

    print("\n===================================")
    print("Testing Date:", test_date)
    print("Motion Lock Probability:", round(prob,3))
    print("===================================\n")


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python test_motion_lock_date.py YYYY-MM-DD")

    else:

        test_date(sys.argv[1])