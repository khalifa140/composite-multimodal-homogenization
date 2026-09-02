"""
Evaluation script: computes R², MAE, RMSE, MAPE, and transverse-isotropy
symmetry deviations for the trained fusion model.

Usage:
    python evaluate.py --model_path ./output/fusion_model.keras \
                       --data_dir /path/to/data
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from models.physics_loss import physics_consistency_loss, mse_only, physics_penalty_only


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fusion model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def compute_metrics(y_true, y_pred, target_cols):
    records = []
    for j, prop in enumerate(target_cols):
        yt, yp = y_true[:, j], y_pred[:, j]
        records.append({
            "Property": prop,
            "R2": round(r2_score(yt, yp), 4),
            "MAE": round(mean_absolute_error(yt, yp), 6),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 6),
            "MAPE_pct": round(
                float(np.mean(np.abs((yt - yp) / np.clip(np.abs(yt), 1e-8, None))) * 100), 4
            ),
        })
    return pd.DataFrame(records).set_index("Property")


def main():
    args = parse_args()
    from tensorflow import keras

    target_cols = ["E11", "E22", "E33", "G12", "G13", "G23", "v12", "v13", "v23"]
    idx = {p: target_cols.index(p) for p in target_cols}

    custom_objects = {
        "physics_consistency_loss": physics_consistency_loss,
        "mse_only": mse_only,
        "physics_penalty_only": physics_penalty_only,
    }

    # Load model and scalers
    print("Loading model...")
    model = keras.models.load_model(args.model_path, custom_objects=custom_objects)

    with open(os.path.join(os.path.dirname(args.model_path), "scaler_tab.pkl"), "rb") as f:
        sc_tab = pickle.load(f)
    with open(os.path.join(os.path.dirname(args.model_path), "scaler_output.pkl"), "rb") as f:
        sc_out = pickle.load(f)

    # Load test data
    X_img = np.load(os.path.join(args.data_dir, "X_img_test.npy"))
    X_tab = np.load(os.path.join(args.data_dir, "X_realistic_test.npy"))
    y_true = np.load(os.path.join(args.data_dir, "y_test.npy"))

    X_tab_sc = sc_tab.transform(X_tab).astype(np.float32)

    # Predict
    print("Predicting...")
    y_pred_sc = model.predict(
        {"image_input": X_img, "tabular_input": X_tab_sc},
        batch_size=args.batch_size, verbose=0,
    )
    y_pred = sc_out.inverse_transform(y_pred_sc)

    # Metrics
    metrics = compute_metrics(y_true, y_pred, target_cols)
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(metrics.to_string())
    print(f"\nMean R²:   {metrics['R2'].mean():.4f}")
    print(f"Mean MAPE: {metrics['MAPE_pct'].mean():.2f}%")

    # Transverse isotropy check
    print("\n" + "=" * 60)
    print("TRANSVERSE ISOTROPY CONSISTENCY")
    print("=" * 60)
    for p1, p2 in [("E22", "E33"), ("G12", "G13"), ("v12", "v13")]:
        i1, i2 = idx[p1], idx[p2]
        rd_true = np.mean(np.abs(y_true[:, i1] - y_true[:, i2]) / np.abs(y_true[:, i1])) * 100
        rd_pred = np.mean(np.abs(y_pred[:, i1] - y_pred[:, i2]) / np.abs(y_pred[:, i1])) * 100
        print(f"  {p1}/{p2}: FEA = {rd_true:.3f}%  |  Predicted = {rd_pred:.3f}%")


if __name__ == "__main__":
    main()
