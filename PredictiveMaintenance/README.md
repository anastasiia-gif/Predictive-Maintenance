# Predictive Maintenance — RUL prediction for Model_00AFC, Model_06BQ, Model_37LB

Deliverables for the AIS predictive-maintenance assignment.

## What it does
Predicts the motor's **Remaining Useful Life (RUL)** from sensor data after every manufactured part
(**UR1**) and raises a **replacement warning 250 parts before** the predicted end of life (**UR2**), for
each of the three machine types.

## Files
| File | Purpose |
|---|---|
| `Assignment1_Predictive_Maintenance.ipynb` | **Deliverable 1** — documented notebook: EDA, cleaning, modelling, evaluation, and the `predict_from_csv` function. Already executed with outputs and plots. |
| `Assignment2_Report.md` | **Deliverable 2** — cost-benefit analysis and recommendations per machine type. |
| `pdm.py` | Shared helper library (loading, cleaning, RUL labels, features, decision). Used by both the notebook and the inference script. |
| `train_all.py` | Trains + validates one model per machine type and saves them to `models/`. |
| `predict_rul.py` | Stand-alone CLI to score a **new** CSV file. |
| `models/` | Trained models (`*_rul_model.joblib`) and validation/economics summaries. |

## How the data works
* Each CSV is a log of 500 000 parts containing ~33 motor cycles.
* A cycle ends when `Status` becomes `Failure` (broke early) or `Maintenance` (scheduled swap at 15 000).
* Within a cycle, RUL counts down to 0 at that event.
* `Motor_Temp` rises as the motor wears — the dominant signal. Readings are noisy, so we use rolling
  (per-cycle) statistics. Injected sensor faults (fixed sentinel values) are cleaned first.

## Results (held-out machines)
| Model | RUL MAE (parts) | UR2 warned in time | Saving / motor cycle |
|---|---:|---:|---:|
| Model_00AFC | ~60 | 100 % | €1 202 |
| Model_06BQ | ~61 | 100 % | €326 |
| Model_37LB | ~87 | 100 % | €2 825 |

(MAE is out of a 15 000-part motor life, i.e. ≈ 0.5 %. Achieved with a 2000-part smoothing window —
see the window-tuning experiment in the notebook.)

## Reproduce
```bash
cd PredictiveMaintenance
python train_all.py          # trains + validates all three models (~4 min)
python predict_rul.py "../Model_37LB/Model_37LB/WB001_data.csv" Model_37LB --out scored.csv
```
Or open and run `Assignment1_Predictive_Maintenance.ipynb` top to bottom.

## Requirements
Python 3.11+, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib` (plus `jupyter`/`nbconvert` to run
the notebook).
