"""Assignment 2 economics: how many unexpected failures does PdM catch, and
what is the resulting monetary value per machine model.

Warning statistics are measured on the held-out TEST machines (no leakage);
the baseline failure rate is a property of the full dataset.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import joblib, pdm

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "models"
RNG = 42

def files_for(model):
    cfg = pdm.MODEL_CONFIG[model]
    return sorted((BASE / cfg["folder"]).glob(cfg["glob"]))

def test_split(model):
    files = files_for(model)
    rng = np.random.default_rng(RNG)
    n_test = max(1, len(files) // 4)
    test = set(rng.choice([f.name for f in files], size=n_test, replace=False))
    return [f for f in files if f.name in test]

rows = []
for model in pdm.MODEL_CONFIG:
    reg = joblib.load(OUT / f"{model}_rul_model.joblib")

    # ---- baseline failure rate over ALL files ----
    nf = nm = 0
    for f in files_for(model):
        s = pd.read_csv(f, usecols=["Status"]).Status
        nf += int((s == "Failure").sum()); nm += int((s == "Maintenance").sum())
    p_fail = nf / (nf + nm)

    # ---- warning behaviour on TEST failure-cycles ----
    caught = fail_cycles = 0
    leads = []
    for f in test_split(model):
        df = pdm.make_training_frame(pdm.load_machine_file(f))
        df["RUL_pred"] = np.clip(reg.predict(df[pdm.FEATURE_COLS]), 0, None)
        df["cyc"] = f.name + "_" + df["cycle"].astype(str)
        df["warning"] = df["RUL_pred"] <= pdm.WARNING_LEAD
        # end status of each cycle (from the terminating event row)
        for cid, c in df.groupby("cyc", sort=False):
            if c["RUL"].min() > 0:
                continue
            end_status = c["Status"].iloc[-1]
            if end_status != "Failure":
                continue
            fail_cycles += 1
            w = c[c["warning"]]
            if len(w) and w["RUL"].iloc[0] > 0:   # warned before the failure row
                caught += 1
                leads.append(w["RUL"].iloc[0])
    catch_rate = caught / fail_cycles if fail_cycles else float("nan")
    mean_lead = float(np.mean(leads)) if leads else float("nan")

    # ---- economics per motor cycle ----
    residual_fail = p_fail * (1 - catch_rate)
    cost_without = pdm.COST_MOTOR + p_fail * pdm.COST_DOWNTIME
    early_life_cost = (mean_lead / pdm.SCHEDULED_INTERVAL) * pdm.COST_MOTOR
    cost_with = pdm.COST_MOTOR + residual_fail * pdm.COST_DOWNTIME + early_life_cost
    saving = cost_without - cost_with

    rows.append({
        "model": model,
        "baseline_failure_rate": round(p_fail, 3),
        "test_failure_cycles": fail_cycles,
        "catch_rate": round(catch_rate, 3),
        "mean_warning_lead": round(mean_lead, 0),
        "residual_failure_rate": round(residual_fail, 3),
        "cost_without_€": round(cost_without, 0),
        "cost_with_€": round(cost_with, 0),
        "saving_per_cycle_€": round(saving, 0),
        "saving_per_1000_motors_€": round(saving * 1000, 0),
    })
    print(json.dumps(rows[-1], indent=2))

pd.DataFrame(rows).to_csv(OUT / "economics_summary.csv", index=False)
(OUT / "economics_summary.json").write_text(json.dumps(rows, indent=2))
print("saved economics_summary")
