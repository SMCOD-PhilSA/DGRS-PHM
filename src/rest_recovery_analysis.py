# Author: Arcee Juan
# DGS Rest Recovery Analysis
#
# Purpose:
# Determine whether disabling (REST) improves state when reactivated.
#
# It answers:
#   When a day is REST and the next day becomes ACTIVE,
#   what state does the antenna return to?

import pandas as pd
from pathlib import Path

# -----------------------------
# File path
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
HEALTH_FILE = BASE_DIR / "daily_health_with_index.csv"

# -----------------------------
# State rules (same hybrid logic)
# -----------------------------
HEALTH_DEGRADING_THRESHOLD = 60.0
DEGRADING_RISK_LEVELS = {"Warning", "Critical"}

def assign_state(active_flag, risk_level, health_index):
    if not bool(active_flag):
        return "REST"

    risk_level = str(risk_level).strip()

    if (risk_level in DEGRADING_RISK_LEVELS) or (float(health_index) < HEALTH_DEGRADING_THRESHOLD):
        return "DEGRADING"

    return "STABLE"

# -----------------------------
# Main analysis
# -----------------------------
def main():

    if not HEALTH_FILE.exists():
        raise FileNotFoundError(f"Missing file: {HEALTH_FILE}")

    df = pd.read_csv(HEALTH_FILE).sort_values("date").reset_index(drop=True)

    df["active_flag"] = df["active_flag"].astype(bool)

    # Assign daily state
    df["state"] = df.apply(
        lambda r: assign_state(r["active_flag"], r["risk_level"], r["health_index"]),
        axis=1
    )

    transitions = []

    for i in range(len(df) - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        # Identify REST -> ACTIVE transitions
        if today["state"] == "REST" and tomorrow["active_flag"] == True:
            transitions.append({
                "date_rest": today["date"],
                "date_reactivated": tomorrow["date"],
                "next_state": tomorrow["state"],
                "next_health_index": tomorrow["health_index"],
                "next_risk_level": tomorrow["risk_level"]
            })

    recovery_df = pd.DataFrame(transitions)

    print("\n=======================================")
    print("DGS REST → ACTIVE Recovery Analysis")
    print("=======================================")

    total = len(recovery_df)
    print(f"\nTotal REST → ACTIVE events found: {total}")

    if total == 0:
        print("\nNo REST → ACTIVE transitions detected.")
        return

    print("\nDistribution of next state after REST:")
    print(recovery_df["next_state"].value_counts())
    print("\nNormalized distribution:")
    print(recovery_df["next_state"].value_counts(normalize=True))

    print("\nAverage health index after reactivation:")
    print(round(recovery_df["next_health_index"].mean(), 2))

    print("\nBreakdown by risk level after reactivation:")
    print(recovery_df["next_risk_level"].value_counts())

    # Optional: save detailed transitions
    output_file = BASE_DIR / "rest_recovery_events.csv"
    recovery_df.to_csv(output_file, index=False)

    print("\nDetailed events saved to:", output_file)


if __name__ == "__main__":
    main()