#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# project root (CSV files are here)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_csv(filename):
    path = PROJECT_ROOT / filename
    if not path.exists():
        raise Exception(f"{filename} not found in root directory")
    return pd.read_csv(path)


def find_time_column(df):
    for c in ["date", "day", "timestamp", "time"]:
        if c in df.columns:
            return c
    return None


class PHMVisualizer(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("DGS PHM Visualization Dashboard")
        self.geometry("1200x800")

        self.build_controls()
        self.build_plot()

    def build_controls(self):

        top = tk.Frame(self)
        top.pack(fill="x", pady=10)

        tk.Label(top, text="Start Date").pack(side="left")

        self.start_date = DateEntry(top)
        self.start_date.pack(side="left", padx=5)

        tk.Label(top, text="End Date").pack(side="left")

        self.end_date = DateEntry(top)
        self.end_date.pack(side="left", padx=5)

        tk.Label(top, text="Plot").pack(side="left", padx=10)

        self.plot_select = ttk.Combobox(
            top,
            width=28,
            values=[
                "Health Index",
                "Motion Lock Probability",
                "Fault Episodes",
                "Servo Oscillation",
                "Servo Stress",
                "Daily Load",
                "Risk Forecast"
            ]
        )

        self.plot_select.current(0)
        self.plot_select.pack(side="left")

        tk.Button(
            top,
            text="Generate Plot",
            command=self.generate_plot
        ).pack(side="left", padx=10)

    def build_plot(self):

        self.fig = plt.Figure(figsize=(10,6))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def filter_dates(self, df):

        time_col = find_time_column(df)

        if time_col is None:
            return df, None

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        start = pd.to_datetime(self.start_date.get_date())
        end = pd.to_datetime(self.end_date.get_date())

        df = df[(df[time_col] >= start) & (df[time_col] <= end)]

        return df, time_col

    def generate_plot(self):

        self.ax.clear()

        choice = self.plot_select.get()

        try:

            if choice == "Health Index":

                df = load_csv("antenna_health_index.csv")
                df, t = self.filter_dates(df)

                if t:
                    self.ax.plot(df[t], df["health_index"])

            elif choice == "Motion Lock Probability":

                df = load_csv("motion_lock_probability.csv")
                df, t = self.filter_dates(df)

                if t:
                    self.ax.plot(df[t], df["motion_lock_probability"])

            elif choice == "Fault Episodes":

                df = load_csv("fault_episodes.csv")

                if "timestamp" in df.columns:
                    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                else:
                    raise Exception("fault_episodes.csv missing timestamp column")

                daily = df.groupby("date").size()

                self.ax.plot(daily.index, daily.values)

            elif choice == "Servo Oscillation":

                df = load_csv("servo_oscillation_daily.csv")
                df, t = self.filter_dates(df)

                if t:
                    self.ax.plot(df[t], df["oscillation_score"])

            elif choice == "Servo Stress":

                df = load_csv("metrics_servo_stress_daily.csv")
                df, t = self.filter_dates(df)

                if t:
                    self.ax.plot(df[t], df["servo_stress"])

            elif choice == "Daily Load":

                df = load_csv("daily_load_forecast.csv")
                df, t = self.filter_dates(df)

                if t:
                    self.ax.plot(df[t], df["load"])

            elif choice == "Risk Forecast":

                df = load_csv("next_7day_risk_forecast.csv")

                if "day" in df.columns and "risk_probability" in df.columns:
                    self.ax.plot(df["day"], df["risk_probability"])
                else:
                    self.ax.plot(df.index, df.iloc[:,1])

        except Exception as e:

            self.ax.text(
                0.5,
                0.5,
                str(e),
                ha="center",
                va="center",
                transform=self.ax.transAxes
            )

        self.ax.set_title(choice)
        self.ax.grid(True)

        self.fig.autofmt_xdate()
        self.canvas.draw()


def main():
    app = PHMVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()