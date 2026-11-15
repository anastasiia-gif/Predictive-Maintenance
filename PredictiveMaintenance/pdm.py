"""
pdm.py — Predictive-Maintenance helper library
================================================

Reusable functions for the RUL (Remaining Useful Life) predictive-maintenance
system for the machines Model_00AFC, Model_06BQ and Model_37LB.

The three machine models share the same sensor schema and the same motor.
The only differences between them are:
  * the file-naming convention of the raw CSV files, and
  * the name of the row-counter column ("nr" or "time").

This module centralises everything that is reused by the Jupyter notebook and
by the stand-alone inference script `predict_rul.py`, so the exact same
cleaning / feature-engineering code is used for training and for scoring a new
CSV file (no train/serve skew).

Design summary
--------------
* A machine CSV is a long log of manufactured parts (one row per part).
* Every ~15 000 parts the motor reaches the end of its life and the row is
  labelled in `Status` as either
      - "Maintenance" : planned replacement at the scheduled interval, or
      - "Failure"     : the motor broke down before the schedule.
  All other rows are labelled "On".
* Each such event marks the end of one *motor cycle*.  Within a cycle the
  Remaining Useful Life (RUL) of a part is simply the number of parts still to
  be produced before the terminating event -> RUL = 0 at the event itself.
* The sensor `Motor_Temp` rises steadily as the motor wears out and is by far
  the strongest predictor of RUL.  Individual readings are very noisy, so we
  smooth the sensors with rolling windows (reset at every cycle boundary).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSORS = ["Ampere", "Motor_Temp", "Vibrations", "Noise"]

# Sentinel values that the data-generation script (`add_errors.py`) injects to
# simulate a sensor read-out fault.  They must be treated as missing data.
SENTINELS = {
    "Ampere": 9.99,
    "Motor_Temp": 999.9999,
    "Vibrations": 999.99,
    "Noise": 99.99,
}

# Rolling-window length (in parts) used to smooth the noisy sensor signal.
# Motor_Temp is very noisy (std ~35 on a slow, monotonic trend), so a long trailing
# window is needed to resolve RUL near end-of-life. W=2000 was selected empirically
# (see the window-tuning experiment in the notebook): raising it from 300 to ~2000
# roughly doubles the UR2 recall without hurting bias, because temperature changes
# slowly enough that the long-window mean stays informative.
WINDOW = 2000
SHORT_WINDOW = 300

# Business assumptions (Assignment 2).
COST_MOTOR = 4500.0            # cost of a replacement motor
COST_DOWNTIME = 3500.0        # *extra* cost of an unexpected failure
SCHEDULED_INTERVAL = 15000    # parts after which the motor is normally replaced
WARNING_LEAD = 250            # UR2: warn this many parts before predicted end

# Where each machine model lives and how its files / counter column are named.
MODEL_CONFIG = {
    "Model_00AFC": {"folder": "Model_00AFC",            "glob": "machine*_data.csv", "counter": "time"},
    "Model_06BQ":  {"folder": "Model_06BQ",             "glob": "Line*_data.csv",    "counter": "nr"},
    "Model_37LB":  {"folder": "Model_37LB/Model_37LB",  "glob": "WB*_data.csv",      "counter": "nr"},
}


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def clean_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Replace injected sensor-fault sentinels by NaN and interpolate them away.

    The faults are rare, isolated single readings, so linear interpolation
    (with forward/backward fill for the edges) reconstructs a plausible value
    without distorting the signal.
    """
    df = df.copy()
    for col, val in SENTINELS.items():
        df.loc[df[col] == val, col] = np.nan
    df[SENSORS] = df[SENSORS].interpolate(limit_direction="both")
    return df


def add_cycle_and_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `cycle` id and the ground-truth `RUL` for every row.

    A cycle ends on the first row whose Status is not "On" (Failure or
    Maintenance).  Within a cycle RUL counts down to 0 at that terminating row.
    A trailing partial cycle (no terminating event at the end of the file) gets
    RUL = NaN because its true end is unknown; such rows are dropped for
    training but kept for inference.
    """
    df = df.copy()
    is_event = (df["Status"] != "On").to_numpy()
    # cycle id increments *after* each event row.
    cycle = np.cumsum(np.concatenate([[0], is_event[:-1]]))
    df["cycle"] = cycle

    rul = np.full(len(df), np.nan)
    event_idx = np.flatnonzero(is_event)
    start = 0
    for e in event_idx:
        rul[start:e + 1] = np.arange(e - start, -1, -1)
        start = e + 1
    df["RUL"] = rul
    df["is_event"] = is_event
    return df


def load_machine_file(path: str | Path) -> pd.DataFrame:
    """Load one raw machine CSV, clean it and attach cycle ids + RUL."""
    df = pd.read_csv(path)
    df = clean_sensors(df)
    df = add_cycle_and_rul(df)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add smoothed rolling sensor features, reset at each motor cycle.

    Rolling statistics are computed *inside* each cycle (grouped by `cycle`) so
    that readings from a previous, worn-out motor never leak into a fresh one.
    This is exactly what happens at inference time on a new CSV, so the feature
    definition is identical for training and scoring.
    """
    df = df.copy()
    g = df.groupby("cycle", sort=False)

    feat = {}
    for s in SENSORS:
        feat[f"{s}_mean"] = g[s].transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        feat[f"{s}_mean_s"] = g[s].transform(lambda x: x.rolling(SHORT_WINDOW, min_periods=1).mean())
    # Temperature is the key wear signal: also its variability and local slope.
    feat["Motor_Temp_std"] = g["Motor_Temp"].transform(lambda x: x.rolling(WINDOW, min_periods=5).std())
    feat["Motor_Temp_slope"] = feat["Motor_Temp_mean"] - g["Motor_Temp"].transform(
        lambda x: x.rolling(WINDOW, min_periods=1).mean().shift(WINDOW)
    )
    fdf = pd.DataFrame(feat, index=df.index)
    out = pd.concat([df, fdf], axis=1)
    return out


FEATURE_COLS = (
    [f"{s}_mean" for s in SENSORS]
    + [f"{s}_mean_s" for s in SENSORS]
    + ["Motor_Temp_std", "Motor_Temp_slope"]
)


def make_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame with a valid RUL target (drops trailing partial cycle)."""
    df = add_features(df)
    df = df.dropna(subset=["RUL"]).copy()
    # Fill the few NaNs produced by std/slope at the very start of a cycle.
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Business logic: warning / decision
# ---------------------------------------------------------------------------

def add_decision(df: pd.DataFrame, pred_col: str = "RUL_pred",
                 lead: int = WARNING_LEAD) -> pd.DataFrame:
    """Add the UR2 replacement warning.

    UR2: raise the warning `lead` parts before the predicted replacement point,
    i.e. as soon as the predicted RUL drops to `lead` or below.
    `replace_now` marks the first part at which the warning becomes active.
    """
    df = df.copy()
    df["warning"] = df[pred_col] <= lead
    # First activation of the warning inside each cycle.
    first = df.groupby("cycle", sort=False)["warning"].transform(
        lambda x: x & (x.cumsum() == 1)
    )
    df["replace_now"] = first
    return df
