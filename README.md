# Predictive Maintenance — Remaining Useful Life Prediction

Machine-learning pipeline that predicts a motor's **Remaining Useful Life (RUL)** from sensor data and raises a replacement warning **250 parts before** the predicted end of life — for three different machine types — plus a cost-benefit analysis of the maintenance strategy.

> Full project in [`PredictiveMaintenance/`](PredictiveMaintenance/) — see its [README](PredictiveMaintenance/README.md) for the file-by-file guide.

## Highlights

- **End-to-end notebook** — EDA, cleaning, feature engineering, modelling, evaluation
- **One trained model per machine type** (`models/*.joblib`), with validation summaries
- **Stand-alone CLI** (`predict_rul.py`) to score new sensor CSVs with the saved models
- **Economics report** — cost-benefit analysis and per-machine-type maintenance recommendations
- Shared helper library (`pdm.py`) used by notebook, training and inference, so there's one source of truth for cleaning and features

## Stack

Python, pandas, scikit-learn, joblib, Jupyter.
