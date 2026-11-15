"""Re-evaluate UR2 warning quality with globally-unique cycle ids (per file)."""
import json, numpy as np, pandas as pd
from pathlib import Path
import joblib, pdm

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "models"
RNG = 42; TRAIN_STRIDE = 15

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
    frames = []
    for f in test_split(model):
        df = pdm.make_training_frame(pdm.load_machine_file(f))
        df["RUL_pred"] = np.clip(reg.predict(df[pdm.FEATURE_COLS]), 0, None)
        # UNIQUE cycle per file, then per-cycle warning
        df["cycle"] = f.name + "_" + df["cycle"].astype(str)
        df = pdm.add_decision(df, "RUL_pred", pdm.WARNING_LEAD)
        frames.append(df)
    t = pd.concat(frames, ignore_index=True)

    leads, missed, n = [], 0, 0
    for _, c in t.groupby("cycle", sort=False):
        if c["RUL"].min() > 0:      # incomplete cycle, skip
            continue
        n += 1
        w = c[c["replace_now"]]
        if len(w) == 0:
            missed += 1
        else:
            leads.append(w["RUL"].iloc[0])
    leads = np.array(leads)
    rows.append({
        "model": model, "cycles_evaluated": n,
        "missed_warnings": missed,
        "warned_in_time_pct": round(100*len(leads)/n, 1),
        "lead_mean": round(float(leads.mean()), 1),
        "lead_median": round(float(np.median(leads)), 1),
        "lead_p05": round(float(np.percentile(leads, 5)), 1),
        "lead_min": round(float(leads.min()), 1),
        "warned_before_failure_pct": round(100*float((leads > 0).mean()), 1),
    })
    print(json.dumps(rows[-1], indent=2))

pd.DataFrame(rows).to_csv(OUT / "ur2_summary.csv", index=False)
(OUT / "ur2_summary.json").write_text(json.dumps(rows, indent=2))
print("saved ur2_summary")
