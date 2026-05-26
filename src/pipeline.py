# Author: Arcee Juan
# Full Predictive Health Pipeline Runner

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "parser.py",
    "fault_episodes.py",
    "daily_features.py",
    "health_index.py",
    "risk_forecast.py"
]


def run_script(script_name):
    script_path = BASE_DIR / script_name
    print(f"\nRunning {script_name}...")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error in {script_name}")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)


def main():
    print("Starting full predictive health pipeline...")

    for script in SCRIPTS:
        run_script(script)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()