# ============================================
# DGS Predictive Health Monitoring GUI
# ============================================

import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
PYTHON_EXEC = sys.executable


# --------------------------------------------
# Utility
# --------------------------------------------

def run_script(script_name):
    subprocess.run([PYTHON_EXEC, str(SRC_DIR / script_name)], cwd=str(BASE_DIR), check=True)


def run_pipeline():
    try:
        run_script("parser.py")
        run_script("fault_episodes.py")
        run_script("daily_features.py")
        run_script("degradation_state_builder.py")
        run_script("load_forecast.py")
        messagebox.showinfo("Pipeline", "Pipeline executed successfully.")
    except Exception as e:
        messagebox.showerror("Pipeline Error", str(e))


# --------------------------------------------
# Load Score Helper (robust)
# --------------------------------------------

def get_load_score():
    load_df = pd.read_csv(BASE_DIR / "weekly_load_forecast.csv")

    for col in load_df.columns:
        if "load" in col.lower() and "score" in col.lower():
            return float(load_df[col].values[0])

    raise ValueError("Load Score column not found in weekly_load_forecast.csv")


# --------------------------------------------
# Forward Projection
# --------------------------------------------

def forward_projection(start_date):
    state_df = pd.read_csv(BASE_DIR / "degradation_state_history.csv", parse_dates=["date"])

    if start_date not in state_df["date"].values:
        raise ValueError("Start date not found in degradation history.")

    D = float(state_df[state_df["date"] == start_date]["degradation_state"].values[0])

    load_score = get_load_score()
    damage_scale = 4.0

    future = []
    for i in range(1, 8):
        d = start_date + timedelta(days=i)
        D = min(100, D + load_score * damage_scale)
        future.append((d.date(), round(D, 3)))

    return pd.DataFrame(future, columns=["date", "predicted_degradation"])


# --------------------------------------------
# Backtest Validation (From–To)
# --------------------------------------------

def backtest_validation(date_from, date_to):
    state_df = pd.read_csv(BASE_DIR / "degradation_state_history.csv", parse_dates=["date"])
    state_df["date"] = state_df["date"].dt.date

    date_range = pd.date_range(date_from, date_to)

    errors = []

    for d in date_range:
        d = d.date()
        if d not in state_df["date"].values:
            continue

        proj = forward_projection(pd.to_datetime(d))
        actual = state_df[state_df["date"].isin(proj["date"])]

        merged = proj.merge(actual, on="date", how="inner")

        if not merged.empty:
            mae = np.mean(np.abs(
                merged["predicted_degradation"] - merged["degradation_state"]
            ))
            errors.append(mae)

    if not errors:
        raise ValueError("No overlapping actual data in selected range.")

    return np.mean(errors)


# --------------------------------------------
# GUI
# --------------------------------------------

def run_forward():
    try:
        start = pd.to_datetime(entry_from.get())
        proj = forward_projection(start)

        output.delete("1.0", tk.END)
        output.insert(tk.END, "7-Day Forward Projection\n\n")
        output.insert(tk.END, proj.to_string(index=False))

    except Exception as e:
        messagebox.showerror("Error", str(e))


def run_backtest():
    try:
        d_from = pd.to_datetime(entry_from.get())
        d_to = pd.to_datetime(entry_to.get())

        mae = backtest_validation(d_from, d_to)

        output.delete("1.0", tk.END)
        output.insert(tk.END, f"Backtest MAE from {d_from.date()} to {d_to.date()}:\n")
        output.insert(tk.END, f"\nMAE = {round(mae,3)}")

    except Exception as e:
        messagebox.showerror("Validation Error", str(e))


root = tk.Tk()
root.title("DGS Predictive Health Monitoring")

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="Run Full Pipeline", command=run_pipeline).grid(row=0, column=0, columnspan=2, pady=5)

tk.Label(frame, text="From (YYYY-MM-DD):").grid(row=1, column=0)
entry_from = tk.Entry(frame, width=15)
entry_from.grid(row=1, column=1)
entry_from.insert(0, datetime.today().strftime("%Y-%m-%d"))

tk.Label(frame, text="To (YYYY-MM-DD):").grid(row=2, column=0)
entry_to = tk.Entry(frame, width=15)
entry_to.grid(row=2, column=1)
entry_to.insert(0, datetime.today().strftime("%Y-%m-%d"))

tk.Button(frame, text="Run 7-Day Forward Forecast", command=run_forward).grid(row=3, column=0, columnspan=2, pady=5)
tk.Button(frame, text="Run Backtest Validation", command=run_backtest).grid(row=4, column=0, columnspan=2, pady=5)

output = tk.Text(root, width=90, height=25)
output.pack(padx=10, pady=10)

root.mainloop()