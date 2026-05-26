# ============================================
# DGS Predictive Health Monitoring Pipeline
# Author: Arcee Juan
# ============================================

import subprocess
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

PYTHON_EXEC = sys.executable


# --------------------------------------------
# Logging
# --------------------------------------------

def log(msg):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


# --------------------------------------------
# Script Runner
# --------------------------------------------

def run(script):

    path = SRC_DIR / script

    if not path.exists():
        raise FileNotFoundError(path)

    log(f"Running {script}")

    result = subprocess.run(
        [PYTHON_EXEC, str(path)],
        cwd=str(BASE_DIR)
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script} failed")

    log(f"Completed {script}")


# --------------------------------------------
# Main Pipeline
# --------------------------------------------

def main():

    log("===================================")
    log("Starting DGS PHM Pipeline")
    log("===================================")

    # ==================================================
    # 1. DATA INGESTION
    # ==================================================
    run("data_ingestion.py")

    # ==================================================
    # 2. EVENT FEATURES
    # ==================================================
    run("fault_episodes.py")
    run("daily_features.py")

    # ==================================================
    # 3. METRICS FEATURES
    # ==================================================
    run("metrics_daily_features.py")

    # ==================================================
    # 4. PHM MODELS
    # ==================================================
    run("motion_lock_probability.py")

    # Health index
    run("phm_health_index.py")

    # Risk forecast
    run("phm_risk_forecast.py")

    # ==================================================
    # 5. PHM STATUS MONITOR
    # ==================================================
    # Generates the operational PHM report
    run("phm_monitor.py")

    log("===================================")
    log("PHM Pipeline Completed Successfully")
    log("===================================")


if __name__ == "__main__":
    main()