import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------
# PATHS
# ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DAILY_FEATURES = PROJECT_ROOT / "metrics_daily_features.csv"
SSI_FILE = PROJECT_ROOT / "metrics_servo_stress_daily.csv"
FAULT_FILE = PROJECT_ROOT / "fault_episodes.csv"

OUT_CORR = PROJECT_ROOT / "fault_metric_correlation.csv"
OUT_HEATMAP = PROJECT_ROOT / "correlation_heatmap.png"
OUT_TIMESERIES = PROJECT_ROOT / "fault_vs_metrics_timeseries.png"


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

def load_data():

    print("Loading datasets...")

    daily = pd.read_csv(DAILY_FEATURES)
    ssi = pd.read_csv(SSI_FILE)
    faults = pd.read_csv(FAULT_FILE)

    daily["date"] = pd.to_datetime(daily["date"])
    ssi["date"] = pd.to_datetime(ssi["date"])

    faults["start_time"] = pd.to_datetime(faults["start_time"])
    faults["date"] = faults["start_time"].dt.date

    faults["date"] = pd.to_datetime(faults["date"])

    return daily, ssi, faults


# ------------------------------------------------
# BUILD FAULT TIMELINE
# ------------------------------------------------

def build_fault_timeline(daily, faults):

    print("Building fault timeline...")

    daily["fault"] = 0

    fault_days = faults["date"].unique()

    daily.loc[daily["date"].isin(fault_days), "fault"] = 1

    return daily


# ------------------------------------------------
# MERGE METRICS
# ------------------------------------------------

def merge_metrics(daily, ssi):

    print("Merging SSI with metrics...")

    df = pd.merge(daily, ssi, on="date", how="left")

    return df


# ------------------------------------------------
# COMPUTE CORRELATIONS
# ------------------------------------------------

def compute_correlations(df):

    print("Computing correlations with faults...")

    numeric_cols = df.select_dtypes(include="number").columns

    correlations = []

    for col in numeric_cols:

        if col == "fault":
            continue

        corr = df[col].corr(df["fault"])

        correlations.append({
            "metric": col,
            "fault_correlation": corr
        })

    corr_df = pd.DataFrame(correlations)

    corr_df = corr_df.sort_values("fault_correlation", ascending=False)

    corr_df.to_csv(OUT_CORR, index=False)

    print("Saved:", OUT_CORR)

    return corr_df


# ------------------------------------------------
# HEATMAP
# ------------------------------------------------

def plot_heatmap(df):

    print("Generating correlation heatmap...")

    numeric_df = df.select_dtypes(include="number")

    corr = numeric_df.corr()

    plt.figure(figsize=(12,10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0
    )

    plt.title("Metric Correlation Matrix")

    plt.tight_layout()

    plt.savefig(OUT_HEATMAP)

    plt.close()

    print("Saved:", OUT_HEATMAP)


# ------------------------------------------------
# TIMESERIES VISUALIZATION
# ------------------------------------------------

def plot_fault_vs_metrics(df):

    print("Generating timeseries plot...")

    plt.figure(figsize=(12,6))

    plt.plot(df["date"], df["servo_stress_index"], label="Servo Stress")

    fault_points = df[df["fault"] == 1]

    plt.scatter(
        fault_points["date"],
        fault_points["servo_stress_index"],
        color="red",
        label="Fault"
    )

    plt.title("Servo Stress vs Fault Events")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()

    plt.savefig(OUT_TIMESERIES)

    plt.close()

    print("Saved:", OUT_TIMESERIES)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    daily, ssi, faults = load_data()

    daily = build_fault_timeline(daily, faults)

    df = merge_metrics(daily, ssi)

    corr_df = compute_correlations(df)

    plot_heatmap(df)

    plot_fault_vs_metrics(df)

    print("\nFault correlation analysis complete.")


if __name__ == "__main__":
    main()