# DGS Antenna Predictive Health Monitoring (PHM) System

## Overview

This project implements a Predictive Health Monitoring (PHM) system for
the Davao Ground Station (DGS) antenna. The system analyzes telemetry
and event logs to assess antenna health, detect degradation trends, and
estimate the probability of motion lock events.

The PHM system processes operational telemetry from the antenna control
system and event logs generated during operations. By computing
engineering features and statistical indicators, the system provides
daily health assessments and short‑term risk forecasts.

The primary objective of the PHM system is to support preventive
maintenance by detecting early signs of servo instability, movement
failures, or abnormal operational conditions.

------------------------------------------------------------------------

# System Architecture

The PHM system consists of five logical layers:

1.  Data ingestion
2.  Feature engineering
3.  PHM modeling
4.  Risk forecasting
5.  Monitoring and reporting

The pipeline flow is:

Telemetry + Event Logs\
↓\
Data Ingestion\
↓\
Feature Engineering\
↓\
PHM Models\
↓\
Risk Forecast\
↓\
Operational PHM Report

------------------------------------------------------------------------

# Data Directory Structure

Telemetry data must be organized by year using the following structure:

    data/
        2022/
            Events/
                *.txt
            Metrics/
                *.csv
        2023/
            Events/
            Metrics/
        2024/
            Events/
            Metrics/
        2025/
            Events/
            Metrics/
        2026/
            Events/
            Metrics/

Events are stored as text log files and metrics contain antenna
telemetry exported as CSV files.

------------------------------------------------------------------------

# Main Pipeline Script

The pipeline is executed using:

    python main.py

The pipeline orchestrates all scripts in the correct order.

Pipeline execution order:

1.  data_ingestion.py\
2.  fault_episodes.py\
3.  daily_features.py\
4.  metrics_daily_features.py\
5.  motion_lock_probability.py\
6.  phm_health_index.py\
7.  phm_risk_forecast.py\
8.  phm_monitor.py

------------------------------------------------------------------------

# Script Descriptions

## data_ingestion.py

Purpose:

Parses new telemetry and event files from the data directory. The script
processes only files that have not previously been parsed.

Key operations:

-   Detects new event files in `data/YYYY/Events`
-   Detects new telemetry files in `data/YYYY/Metrics`
-   Extracts relevant event lines
-   Extracts selected telemetry columns
-   Appends parsed data to master datasets

Outputs:

events_master.csv\
metrics_master.csv

------------------------------------------------------------------------

## fault_episodes.py

Purpose:

Identifies fault episodes from event logs. Fault episodes represent
periods where antenna warnings or faults occurred.

The script scans event logs and groups events into episodes based on
timestamps.

Output:

fault_episodes.csv

------------------------------------------------------------------------

## daily_features.py

Purpose:

Converts event‑based fault data into daily indicators used by the PHM
models.

Typical computed features include:

-   number of faults per day
-   duration of fault episodes
-   frequency of alarms

Output:

events_daily_features.csv

------------------------------------------------------------------------

## metrics_daily_features.py

Purpose:

Processes telemetry data and extracts daily engineering indicators from
antenna telemetry.

Key telemetry signals used:

-   axis current
-   following error
-   axis velocity
-   antenna azimuth velocity
-   antenna elevation velocity
-   antenna brake state
-   cabinet temperature

Daily statistics computed:

Mean value Standard deviation Maximum value Signal variability Error
spike counts

Important calculated indicators:

Tracking Ratio\
The fraction of telemetry samples during which the antenna is moving.

Failed Movement Attempts\
Detected when the antenna brake toggles from low to high immediately
after movement.

Error Spikes\
High following error values indicating servo instability.

Output:

metrics_daily_features.csv

------------------------------------------------------------------------

## motion_lock_probability.py

Purpose:

Estimates the probability that the antenna will experience a motion lock
condition.

The model uses historical telemetry patterns associated with past motion
lock events.

Input features include:

-   following error statistics
-   velocity variability
-   current spikes
-   tracking activity
-   failed movement attempts

The model outputs a probability value between 0 and 1.

Output:

motion_lock_probability.csv

------------------------------------------------------------------------

## phm_health_index.py

Purpose:

Computes an overall antenna health score.

The health index combines multiple normalized indicators including:

-   error spikes
-   current variation
-   temperature anomalies
-   failed movements

Example formula:

Health Index = 0.45 × failed movement score + 0.30 × error spike score +
0.15 × current anomaly score + 0.10 × temperature anomaly score

Health status classification:

Healthy: index \< 0.30\
Warning: 0.30 ≤ index \< 0.60\
Critical: index ≥ 0.60

Output:

antenna_health_index.csv

------------------------------------------------------------------------

## phm_risk_forecast.py

Purpose:

Forecasts motion lock risk over the next 24 to 72 hours.

The forecast combines:

Current motion lock probability Short‑term degradation trends Movement
failure trends Error spike trends

Forecast values include:

24‑hour risk\
48‑hour risk\
72‑hour risk

Output:

phm_risk_forecast.csv

------------------------------------------------------------------------

## phm_monitor.py

Purpose:

Generates the operational PHM status report used by operators.

The script summarizes:

Antenna health index Motion lock probability Risk forecast Operational
indicators

Example output:

DGS ANTENNA PHM STATUS

Date: YYYY‑MM‑DD

Antenna Health\
Health Index: 0.112\
Status: HEALTHY

Motion Lock Risk\
Probability: 0.051\
Risk Level: LOW

Risk Forecast\
24h risk: 0.043\
48h risk: 0.048\
72h risk: 0.051

Operational Indicators\
Tracking ratio: 0.058\
Failed movement attempts: 0\
Error spikes: 1071

------------------------------------------------------------------------

# Running the PHM Pipeline

To execute the full PHM system:

    python main.py

The pipeline will:

1.  Parse new telemetry and event files
2.  Update feature datasets
3.  Compute health and risk models
4.  Generate the operational PHM report

------------------------------------------------------------------------

# Historical Backtracking

Backtracking evaluates how the PHM system would have predicted failures
using historical data.

To evaluate a specific date:

    python src/phm_monitor.py YYYY-MM-DD

Example:

    python src/phm_monitor.py 2025-06-05

The script will compute the PHM status using data up to that date.

------------------------------------------------------------------------

# Motion Lock Event Validation

Known motion lock events:

2022‑11‑25\
2023‑04‑05\
2025‑06‑06\
2026‑03‑04

To evaluate model prediction accuracy, run PHM on the days preceding
these events.

Example:

    python src/phm_monitor.py 2025-06-01
    python src/phm_monitor.py 2025-06-03
    python src/phm_monitor.py 2025-06-05

A good PHM system should show a rising risk trend before the failure.

Example progression:

5 days before failure → LOW risk\
3 days before failure → MEDIUM risk\
1 day before failure → HIGH risk

------------------------------------------------------------------------

# Engineering Indicators Used

The PHM models rely on several physical indicators derived from
telemetry.

Tracking Ratio\
Indicates antenna activity and operational confidence.

Following Error\
Measures servo tracking accuracy.

Velocity Variability\
Indicates servo oscillation.

Current Variation\
Detects mechanical load changes.

Brake State Transitions\
Detects failed movement attempts.

Temperature Metrics\
Detect thermal anomalies.

------------------------------------------------------------------------

# PHM System Purpose

The PHM system aims to:

Detect early signs of antenna degradation\
Predict motion lock conditions\
Provide maintenance planning indicators\
Improve ground station operational reliability

------------------------------------------------------------------------

# Author
Arcee T. Juan :)
DGS Predictive Health Monitoring System
