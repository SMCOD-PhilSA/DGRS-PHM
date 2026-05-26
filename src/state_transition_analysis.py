# Author: Arcee Juan
# DGS State Transition Analysis (Stable / Degrading / Rest)
#
# Inputs:
#   - daily_health_with_index.csv   (must include: date, active_flag, risk_level, health_index)
#   - daily_load_forecast.csv       (must include: date, load_score_day)  [optional but recommended]
#
# Outputs:
#   - state_transition_matrix.csv
#   - state_transition_counts.csv
#   - state_transition_by_load_bucket.csv
#   - next_7day_state_forecast.csv

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

HEALTH_FILE = BASE_DIR / "daily_health_with_index.csv"
LOAD_FILE = BASE_DIR / "daily_load_forecast.csv"

OUT_MATRIX = BASE_DIR / "state_transition_matrix.csv"
OUT_COUNTS = BASE_DIR / "state_transition_counts.csv"
OUT_BY_LOAD = BASE_DIR / "state_transition_by_load_bucket.csv"
OUT_NEXT7 = BASE_DIR / "next_7day_state_forecast.csv"

# -----------------------------
# State definitions (Hybrid)
# -----------------------------
HEALTH_DEGRADING_THRESHOLD = 60.0
DEGRADING_RISK_LEVELS = {"Warning", "Critical"}

STATES = ["STABLE", "DEGRADING", "REST"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}


def assign_state(active_flag: bool, risk_level: str, health_index: float) -> str:
    if not bool(active_flag):
        return "REST"

    risk_level = str(risk_level).strip()
    if (risk_level in DEGRADING_RISK_LEVELS) or (float(health_index) < HEALTH_DEGRADING_THRESHOLD):
        return "DEGRADING"

    return "STABLE"


def transition_matrix_from_series(state_series: pd.Series):
    counts = np.zeros((len(STATES), len(STATES)), dtype=int)

    s = state_series.tolist()
    for i in range(len(s) - 1):
        a = s[i]
        b = s[i + 1]
        if a not in STATE_TO_IDX or b not in STATE_TO_IDX:
            continue
        counts[STATE_TO_IDX[a], STATE_TO_IDX[b]] += 1

    probs = counts.astype(float)
    for i in range(len(STATES)):
        row_sum = probs[i].sum()
        if row_sum > 0:
            probs[i] = probs[i] / row_sum

    return counts, probs


def bucket_load(x: float) -> str:
    # Simple interpretable buckets
    if pd.isna(x):
        return "unknown"
    x = float(x)
    if x < 0.30:
        return "low"
    if x < 0.60:
        return "moderate"
    return "high"


def simulate_next_7_days(
    start_state: str,
    transition_probs: np.ndarray,
    load_forecast: pd.DataFrame,
    force_rest_dates: set[str] | None = None,
    seed: int = 7,
):
    """
    Simulate a probable next 7-day state path using the learned transition matrix.
    Optionally force specific dates to REST (your operational "disable antenna" decision).
    """
    rng = np.random.default_rng(seed)
    force_rest_dates = force_rest_dates or set()

    cur = start_state
    path = []

    for _, row in load_forecast.iterrows():
        d = str(row["date"])

        if d in force_rest_dates:
            nxt = "REST"
        else:
            # Sample next state based on current state's transition probabilities
            p = transition_probs[STATE_TO_IDX[cur]]
            # If row has no data (all zeros), default to staying
            if np.allclose(p.sum(), 0):
                nxt = cur
            else:
                nxt = rng.choice(STATES, p=p)

            # Practical constraint:
            # If we are REST today and not forced REST tomorrow, allow leaving REST by sampling normally
            # (already handled)

        path.append({
            "date": d,
            "load_score_day": float(row.get("load_score_day", np.nan)),
            "state": nxt
        })
        cur = nxt

    return pd.DataFrame(path)


def main():
    if not HEALTH_FILE.exists():
        raise FileNotFoundError(f"Missing: {HEALTH_FILE}")

    df = pd.read_csv(HEALTH_FILE).sort_values("date").reset_index(drop=True)
    df["active_flag"] = df["active_flag"].astype(bool)

    # Assign daily state
    df["state"] = df.apply(
        lambda r: assign_state(r["active_flag"], r["risk_level"], r["health_index"]),
        axis=1
    )

    # Transition matrix
    counts, probs = transition_matrix_from_series(df["state"])

    # Save matrix + counts
    counts_df = pd.DataFrame(counts, index=STATES, columns=STATES)
    probs_df = pd.DataFrame(probs, index=STATES, columns=STATES)

    counts_df.to_csv(OUT_COUNTS)
    probs_df.to_csv(OUT_MATRIX)

    print("\n==============================")
    print("DGS State Transition Analysis")
    print("==============================")
    print("\nTransition counts:")
    print(counts_df)
    print("\nTransition probabilities:")
    print(probs_df)

    # Load-conditional transitions (optional)
    by_load_rows = []
    if LOAD_FILE.exists():
        load_df = pd.read_csv(LOAD_FILE).copy()
        load_df["date"] = load_df["date"].astype(str)

        df2 = df.copy()
        df2["date"] = df2["date"].astype(str)

        merged = df2.merge(load_df[["date", "load_score_day"]], on="date", how="left")
        merged["load_bucket"] = merged["load_score_day"].apply(bucket_load)

        # Compute transitions within each bucket (from day t bucket)
        for bucket, g in merged.groupby("load_bucket"):
            # Need sequential transitions on original ordering; keep ordering by date
            g = g.sort_values("date").reset_index(drop=True)
            c, p = transition_matrix_from_series(g["state"])
            p_df = pd.DataFrame(p, index=STATES, columns=STATES)
            for from_s in STATES:
                for to_s in STATES:
                    by_load_rows.append({
                        "load_bucket": bucket,
                        "from_state": from_s,
                        "to_state": to_s,
                        "prob": float(p_df.loc[from_s, to_s]),
                        "count": int(c[STATE_TO_IDX[from_s], STATE_TO_IDX[to_s]])
                    })

        out_by_load = pd.DataFrame(by_load_rows)
        out_by_load.to_csv(OUT_BY_LOAD, index=False)

        print("\nSaved transition-by-load buckets to:", OUT_BY_LOAD)
    else:
        print("\nNote: daily_load_forecast.csv not found; skipping load-bucket transitions.")

    # Next 7-day forecast simulation
    if LOAD_FILE.exists():
        load_df = pd.read_csv(LOAD_FILE).copy()
        load_df["date"] = load_df["date"].astype(str)

        start_state = df["state"].iloc[-1]
        print("\nCurrent state:", start_state)

        # Scenario 1: always active (no forced rest)
        scenario1 = simulate_next_7_days(
            start_state=start_state,
            transition_probs=probs,
            load_forecast=load_df,
            force_rest_dates=set(),
            seed=7
        )
        scenario1["scenario"] = "always_active"

        # Scenario 2: proactive rest on the highest-load day
        # (Example policy: rest on max load_score_day)
        max_day = load_df.loc[load_df["load_score_day"].idxmax(), "date"]
        scenario2 = simulate_next_7_days(
            start_state=start_state,
            transition_probs=probs,
            load_forecast=load_df,
            force_rest_dates={str(max_day)},
            seed=7
        )
        scenario2["scenario"] = f"rest_on_max_load_day_{max_day}"

        out_next7 = pd.concat([scenario1, scenario2], ignore_index=True)
        out_next7.to_csv(OUT_NEXT7, index=False)

        print("\n==============================")
        print("Next 7-Day State Forecast (Simulated)")
        print("==============================")
        print(out_next7)

        print("\nSaved next 7-day forecast to:", OUT_NEXT7)
        print("Saved transition matrix to:", OUT_MATRIX)
        print("Saved transition counts to:", OUT_COUNTS)
    else:
        print("\nNote: daily_load_forecast.csv not found; skipping next 7-day simulation.")

    print("\nDone.")

    


if __name__ == "__main__":
    main()