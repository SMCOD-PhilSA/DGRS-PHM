# ============================================
# Motion Lock Probability GUI (Calendar Version)
# ============================================

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = BASE_DIR / "metrics_daily_features.csv"
STRESS_FILE = BASE_DIR / "metrics_servo_stress_daily.csv"
MODEL_FILE = BASE_DIR / "motion_lock_model.pkl"


# ============================================
# LOAD DATA
# ============================================

def load_data():

    features = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    stress = pd.read_csv(STRESS_FILE, parse_dates=["date"])

    df = pd.merge(features, stress, on="date", how="left")

    df = df.sort_values("date")

    stress_col = df["servo_stress_index"]

    df["stress_slope"] = stress_col.diff()
    df["stress_volatility"] = stress_col.rolling(5).std()

    df = df.fillna(0)

    return df


# ============================================
# FEATURE LIST
# ============================================

FEATURE_COLUMNS = [
    "servo_stress_index",
    "stress_slope",
    "stress_volatility",
    "Upper/X following error_mean",
    "Upper/X following error_max",
    "Lower/Y following error_mean",
    "Lower/Y following error_max",
    "Upper axis current_mean",
    "Lower axis current_mean",
    "Upper/X velocity_mean",
    "Lower/Y velocity_mean",
    "Cabinet temperature_mean"
]


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_probability():

    selected_date = cal.get_date()

    row = df[df["date"] == pd.to_datetime(selected_date)]

    if row.empty:

        messagebox.showerror(
            "Date Not Found",
            "No telemetry data for this date."
        )
        return

    X = row[FEATURE_COLUMNS]

    prob = model.predict_proba(X)[0][1]

    p24 = prob * 0.5
    p48 = prob * 0.8
    p72 = prob

    result_text.set(
        f"\nMotion Lock Risk for {selected_date}\n\n"
        f"24 hours : {p24*100:.1f}%\n"
        f"48 hours : {p48*100:.1f}%\n"
        f"72 hours : {p72*100:.1f}%"
    )


# ============================================
# LOAD DATA + MODEL
# ============================================

print("Loading telemetry data...")

df = load_data()

model = joblib.load(MODEL_FILE)


# ============================================
# GUI
# ============================================

root = tk.Tk()
root.title("DGS Motion Lock Predictor")
root.geometry("420x320")

title = ttk.Label(
    root,
    text="DGS Motion Lock Predictor",
    font=("Segoe UI", 16)
)

title.pack(pady=15)


date_label = ttk.Label(
    root,
    text="Select Telemetry Date",
    font=("Segoe UI", 10)
)

date_label.pack()


# Calendar widget
cal = DateEntry(
    root,
    width=16,
    background="darkblue",
    foreground="white",
    borderwidth=2,
    date_pattern="yyyy-mm-dd"
)

cal.pack(pady=8)


predict_button = ttk.Button(
    root,
    text="Predict Motion Lock Risk",
    command=predict_probability
)

predict_button.pack(pady=10)


result_text = tk.StringVar()

result_label = ttk.Label(
    root,
    textvariable=result_text,
    font=("Segoe UI", 11),
    justify="center"
)

result_label.pack(pady=15)


root.mainloop()