# Author: Arcee Juan
# DGS State Transition Monte Carlo Forecast with REST->ACTIVE Recovery Conditioning

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

HEALTH_FILE = BASE_DIR / "daily_health_with_index.csv"
LOAD_FILE = BASE_DIR / "daily_load_forecast.csv"

OUT_MATRIX = BASE_DIR / "state_transition_matrix.csv"
OUT_COUNTS = BASE_DIR / "state_transition_counts.csv"
OUT_RECOVERY = BASE_DIR / "rest_recovery_params.csv"
OUT_MC = BASE_DIR / "next_7day_state_montecarlo.csv"

# -----------------------------
# State definitions (Hybrid)
# -----------------------------
HEALTH_DEGRADING_THRESHOLD = 60.0
DEGRADING_RISK_LEVELS = {"Warning", "Critical"}

STATES = ["STABLE", "DEGRADING", "REST"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}

# Empirical recovery (from your data)
RECOVERY_P_STABLE = 18 / 27  # 0.6667
RECOVERY_P_DEGRADING = 9 / 27  # 0.3333

# Monte Carlo settings
N_SIMS = 2000
SEED = 42

# Operational policy: if not forced rest, we assume "active" during the forecast window.
ASSUME_ACTIVE_DURING_FORECAST = True


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


def simulate_one_path(
    start_state: str,
    trans_probs: np.ndarray,
    forecast_dates: list[str],
    forced_rest_dates: set[str],
    rng: np.random.Generator,
):
    """
    Simulate one 7-day path.
    Rule: REST->ACTIVE uses empirical recovery probabilities.
    Otherwise uses transition matrix.
    """

    cur = start_state
    path = []

    for d in forecast_dates:

        tomorrow_active = True
        if d in forced_rest_dates:
            # forced operational decision: disable today
            nxt = "REST"
            tomorrow_active = False  # today is inactive; affects conditioning next day via 'cur'
        else:
            # Normal day: assume active for the forecast window
            # We do NOT force REST unless policy says so.
            # We'll sample next state based on current state's row.
            nxt = None

            # Conditional recovery:
            # If current is REST and today is not forced REST, then today is ACTIVE (reactivation)
            if cur == "REST" and ASSUME_ACTIVE_DURING_FORECAST:
                # Reactivation event
                nxt = rng.choice(
                    ["STABLE", "DEGRADING"],
                    p=[RECOVERY_P_STABLE, RECOVERY_P_DEGRADING]
                )
            else:
                p = trans_probs[STATE_TO_IDX[cur]]
                if np.allclose(p.sum(), 0):
                    nxt = cur
                else:
                    nxt = rng.choice(STATES, p=p)

                # Practical constraint:
                # If we assume active during forecast, we should not spontaneously go REST
                # unless forced by policy.
                if ASSUME_ACTIVE_DURING_FORECAST and nxt == "REST":
                    # Re-sample among non-REST states proportional to their probs
                    non_rest = ["STABLE", "DEGRADING"]
                    p2 = np.array([p[STATE_TO_IDX["STABLE"]], p[STATE_TO_IDX["DEGRADING"]]], dtype=float)
                    if p2.sum() <= 0:
                        nxt = cur if cur != "REST" else "DEGRADING"
                    else:
                        p2 = p2 / p2.sum()
                        nxt = rng.choice(non_rest, p=p2)

        path.append(nxt)
        cur = nxt

    return path


def summarize_paths(paths: np.ndarray, forecast_dates: list[str], scenario_name: str):
    """
    paths: shape (n_sims, horizon)
    """
    summary_rows = []

    for j, d in enumerate(forecast_dates):
        col = paths[:, j]
        total = len(col)
        p_stable = np.sum(col == "STABLE") / total
        p_degrading = np.sum(col == "DEGRADING") / total
        p_rest = np.sum(col == "REST") / total

        summary_rows.append({
            "scenario": scenario_name,
            "date": d,
            "p_stable": p_stable,
            "p_degrading": p_degrading,
            "p_rest": p_rest
        })

    # Aggregate metrics
    degrading_any = np.mean(np.any(paths == "DEGRADING", axis=1))
    degrading_days_expected = float(np.mean(np.sum(paths == "DEGRADING", axis=1)))
    stable_days_expected = float(np.mean(np.sum(paths == "STABLE", axis=1)))
    rest_days_expected = float(np.mean(np.sum(paths == "REST", axis=1)))

    agg = {
        "scenario": scenario_name,
        "p_degrading_at_least_once_7d": float(degrading_any),
        "expected_degrading_days_7d": degrading_days_expected,
        "expected_stable_days_7d": stable_days_expected,
        "expected_rest_days_7d": rest_days_expected
    }

    return pd.DataFrame(summary_rows), agg


def main():

    if not HEALTH_FILE.exists():
        raise FileNotFoundError(f"Missing file: {HEALTH_FILE}")
    if not LOAD_FILE.exists():
        raise FileNotFoundError(f"Missing file: {LOAD_FILE}")

    df = pd.read_csv(HEALTH_FILE).sort_values("date").reset_index(drop=True)
    df["active_flag"] = df["active_flag"].astype(bool)

    df["state"] = df.apply(
        lambda r: assign_state(r["active_flag"], r["risk_level"], r["health_index"]),
        axis=1
    )

    # Transition matrix from history
    counts, probs = transition_matrix_from_series(df["state"])
    counts_df = pd.DataFrame(counts, index=STATES, columns=STATES)
    probs_df = pd.DataFrame(probs, index=STATES, columns=STATES)
    counts_df.to_csv(OUT_COUNTS)
    probs_df.to_csv(OUT_MATRIX)

    # Save recovery params for traceability
    rec_df = pd.DataFrame([{
        "recovery_p_stable": RECOVERY_P_STABLE,
        "recovery_p_degrading": RECOVERY_P_DEGRADING,
        "n_events": 27
    }])
    rec_df.to_csv(OUT_RECOVERY, index=False)

    start_state = df["state"].iloc[-1]
    print("\nCurrent state:", start_state)

    load_df = pd.read_csv(LOAD_FILE).copy()
    load_df["date"] = load_df["date"].astype(str)
    forecast_dates = load_df["date"].tolist()

    # Policy definitions
    max_load_day = load_df.loc[load_df["load_score_day"].idxmax(), "date"]

    policies = {
        "always_active": set(),
        f"rest_1day_on_maxload_{max_load_day}": {str(max_load_day)},
        f"rest_2days_from_maxload_{max_load_day}": {
            str(max_load_day),
            str((pd.to_datetime(max_load_day) + pd.Timedelta(days=1)).date())
        }
    }

    rng = np.random.default_rng(SEED)

    all_daily_summaries = []
    agg_rows = []

    for name, forced_rest_dates in policies.items():

        sims = []
        for _ in range(N_SIMS):
            path = simulate_one_path(
                start_state=start_state,
                trans_probs=probs,
                forecast_dates=forecast_dates,
                forced_rest_dates=forced_rest_dates,
                rng=rng
            )
            sims.append(path)

        paths = np.array(sims, dtype=object)
        daily_summary, agg = summarize_paths(paths, forecast_dates, name)

        all_daily_summaries.append(daily_summary)
        agg_rows.append(agg)

    daily_out = pd.concat(all_daily_summaries, ignore_index=True)
    agg_out = pd.DataFrame(agg_rows)

    daily_out.to_csv(OUT_MC, index=False)

    print("\n==============================")
    print("Monte Carlo 7-Day State Forecast (Summary)")
    print("==============================")
    print("\nAggregate comparison of policies:")
    print(agg_out)

    print("\nDaily probabilities (first few rows):")
    print(daily_out.head(14))

    print("\nSaved daily Monte Carlo summary to:", OUT_MC)
    print("Saved transition matrix to:", OUT_MATRIX)
    print("Saved transition counts to:", OUT_COUNTS)
    print("Saved recovery params to:", OUT_RECOVERY)
    print("\nDone.")


if __name__ == "__main__":
    main()