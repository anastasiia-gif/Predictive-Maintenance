"""
train_all.py — train + validate one RUL model per machine type and persist it.

Methodology (AIS workflow):
  1. Split the machines of a model into train / test *by file* (a whole machine
     is either train or test) so no cycle is seen during both fitting and
     evaluation -> honest generalisation estimate.
  2. Sub-sample rows for fitting (every K-th part) to keep it fast; the signal
     is smooth so this loses nothing.
  3. Metrics: RUL MAE/RMSE (UR1) and the UR2 warning lead-time + safety
     (does the warning fire *before* the real end, and how early).
  4. Persist the fitted model with joblib for reuse by the inference script.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pdm

BASE = Path(__file__).resolve().parent.parent   # the AIS folder
OUT = Path(__file__).resolve().parent / "models"
OUT.mkdir(exist_ok=True)

TRAIN_STRIDE = 15   # keep every 15th part for fitting
RNG = 42


def files_for(model: str):
    cfg = pdm.MODEL_CONFIG[model]
    return sorted((BASE / cfg["folder"]).glob(cfg["glob"]))


def build(model: str, files):
    frames = []
    for f in files:
        df = pdm.load_machine_file(f)
        df = pdm.make_training_frame(df)
        df["file"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def evaluate_ur2(test_df: pd.DataFrame) -> dict:
    """Per-cycle UR2 warning analysis on the held-out machines."""
    test_df = pdm.add_decision(test_df, "RUL_pred", pdm.WARNING_LEAD)
    leads, missed, false_frac = [], 0, []
    grp = test_df.groupby(["file", "cycle"], sort=False)
    n_cycles = 0
    for _, c in grp:
        # only evaluate cycles that actually reach the end (RUL==0 present)
        if c["RUL"].min() > 0:
            continue
        n_cycles += 1
        warned = c[c["replace_now"]]
        if len(warned) == 0:
            missed += 1
            continue
        # actual parts remaining at the moment the warning first fires
        lead = warned["RUL"].iloc[0]
        leads.append(lead)
    leads = np.array(leads)
    return {
        "n_cycles": int(n_cycles),
        "missed_warnings": int(missed),
        "warning_lead_mean": float(np.mean(leads)) if len(leads) else float("nan"),
        "warning_lead_p05": float(np.percentile(leads, 5)) if len(leads) else float("nan"),
        "warning_lead_p95": float(np.percentile(leads, 95)) if len(leads) else float("nan"),
        "warned_before_end_pct": float(100.0 * np.mean(leads > 0)) if len(leads) else float("nan"),
    }


def run_model(model: str) -> dict:
    t0 = time.time()
    files = files_for(model)
    rng = np.random.default_rng(RNG)
    n_test = max(1, len(files) // 4)
    test_files = set(rng.choice([f.name for f in files], size=n_test, replace=False))
    train_files = [f for f in files if f.name not in test_files]
    test_files_l = [f for f in files if f.name in test_files]

    print(f"\n===== {model} =====  {len(files)} machines "
          f"({len(train_files)} train / {len(test_files_l)} test)")

    train = build(model, train_files)
    test = build(model, test_files_l)

    Xtr = train[pdm.FEATURE_COLS].iloc[::TRAIN_STRIDE]
    ytr = train["RUL"].iloc[::TRAIN_STRIDE]

    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=8,
        l2_regularization=1.0, random_state=RNG,
    )
    reg.fit(Xtr, ytr)

    test = test.copy()
    test["RUL_pred"] = np.clip(reg.predict(test[pdm.FEATURE_COLS]), 0, None)

    mae = mean_absolute_error(test["RUL"], test["RUL_pred"])
    rmse = mean_squared_error(test["RUL"], test["RUL_pred"]) ** 0.5
    # accuracy near end of life (RUL <= 1000) is what matters for the decision
    near = test[test["RUL"] <= 1000]
    mae_near = mean_absolute_error(near["RUL"], near["RUL_pred"])

    ur2 = evaluate_ur2(test)

    joblib.dump(reg, OUT / f"{model}_rul_model.joblib")

    res = {
        "model": model, "n_machines": len(files),
        "n_train_rows": int(len(Xtr)),
        "RUL_MAE": round(mae, 1), "RUL_RMSE": round(rmse, 1),
        "RUL_MAE_last1000": round(mae_near, 1),
        **ur2, "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    results = [run_model(m) for m in pdm.MODEL_CONFIG]
    pd.DataFrame(results).to_csv(OUT / "validation_summary.csv", index=False)
    (OUT / "validation_summary.json").write_text(json.dumps(results, indent=2))
    print("\nSaved models + validation_summary to", OUT)
