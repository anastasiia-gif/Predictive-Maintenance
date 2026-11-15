"""
predict_rul.py — stand-alone inference for the predictive-maintenance system.

Reads a NEW machine CSV, predicts the Remaining Useful Life (RUL) after every
part (UR1) and issues a replacement warning 250 parts before the predicted end
of life (UR2). Uses the model trained by `train_all.py` (saved in ./models).

Usage:
    python predict_rul.py <path_to_new.csv> <Model_00AFC|Model_06BQ|Model_37LB> [--out result.csv]

Example:
    python predict_rul.py "../Model_37LB/Model_37LB/WB001_data.csv" Model_37LB --out scored.csv
"""
import argparse
import numpy as np
import joblib
from pathlib import Path

import pdm

MODELS_DIR = Path(__file__).resolve().parent / "models"


def predict_from_csv(csv_path, model_name, lead=pdm.WARNING_LEAD):
    """Return a DataFrame with per-part RUL prediction (UR1) and warning (UR2)."""
    model_file = MODELS_DIR / f"{model_name}_rul_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(
            f"No trained model for {model_name}. Run train_all.py first ({model_file})."
        )
    model = joblib.load(model_file)

    df = pdm.load_machine_file(csv_path)          # clean + cycle ids
    df = pdm.add_features(df)
    df[pdm.FEATURE_COLS] = df[pdm.FEATURE_COLS].fillna(0.0)
    df["RUL_pred"] = np.clip(model.predict(df[pdm.FEATURE_COLS]), 0, None).round(0)
    df = pdm.add_decision(df, "RUL_pred", lead)
    return df


def main():
    ap = argparse.ArgumentParser(description="Predict motor RUL and replacement warnings for a CSV.")
    ap.add_argument("csv", help="path to the new machine CSV")
    ap.add_argument("model", choices=list(pdm.MODEL_CONFIG), help="machine model type")
    ap.add_argument("--lead", type=int, default=pdm.WARNING_LEAD,
                    help="warn this many parts before predicted end (default 250)")
    ap.add_argument("--out", help="optional path to write the full scored CSV")
    args = ap.parse_args()

    df = predict_from_csv(args.csv, args.model, args.lead)

    keep = [c for c in ["nr", "time"] if c in df.columns]
    cols = keep + pdm.SENSORS + ["RUL_pred", "warning", "replace_now"]
    n_warn = int(df["replace_now"].sum())
    print(f"Scored {len(df):,} parts from {Path(args.csv).name} with model {args.model}.")
    print(f"Replacement warnings raised: {n_warn}")
    if n_warn:
        print("\nWarning trigger points (first part of each warning, per motor cycle):")
        print(df.loc[df["replace_now"], keep + ["Motor_Temp", "RUL_pred"]].to_string(index=False))

    if args.out:
        df[cols].to_csv(args.out, index=False)
        print(f"\nFull per-part result written to {args.out}")


if __name__ == "__main__":
    main()
