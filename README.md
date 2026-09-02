# Multimodal Deep Learning for Elastic Property Prediction of Unidirectional Composites

Physics-consistent CNN–MLP fusion framework for predicting the full 3D elastic
stiffness tensor of unidirectional fiber-reinforced polymer composites from
microstructural cross-sectional images and constituent material properties.

## Overview

| Component | Description |
|---|---|
| **Image branch** | Lightweight 3-stage Micro-ResNet (~720K params) |
| **Tabular branch** | MLP processing 8 constituent elastic properties |
| **Fusion** | Feature-level concatenation → regression head |
| **Physics loss** | Transverse-isotropy symmetry regularization |
| **Training** | Two-phase: frozen warm-up → end-to-end fine-tuning |
| **Outputs** | 9 elastic constants (E11, E22, E33, G12, G13, G23, ν12, ν13, ν23) |

## Architecture
Input: 384×384 RVE image Input: 8 constituent properties
│ │
Micro-ResNet MLP Branch
(3 residual stages) (128 → 64 → 32)
│ │
GAP → 128-dim 32-dim
│ │
└────── Concatenate (160-dim) ─────┘
│
Fusion Head
(128 → 64 → 9)
│
9 Elastic Constants

text


## Quick Start

### Installation

```bash
pip install -r requirements.txt
Inference Demo
See demo/demo_inference.ipynb for a complete walkthrough that loads
the trained model and predicts elastic properties from a sample RVE image
and constituent property vector.

Training
Bash

python train.py --data_dir /path/to/dataset --epochs 250
Evaluation
Bash

python evaluate.py --model_path /path/to/model --test_dir /path/to/test
Dataset
The training dataset consists of 3,000 RVE samples generated via finite
element homogenization (ABAQUS) under periodic boundary conditions:

Fiber type: T300 carbon fiber (transversely isotropic)
Matrix: Epoxy 7901 (isotropic)
Vf range: 0.29–0.71 (continuous, Monte Carlo)
Constituent variation: 8 elastic parameters independently sampled
Image resolution: 384×384 grayscale
The full dataset is available from the corresponding author upon request.

Key Results
Model	R²	MAPE
MLP (constituents only)	0.393	14.20%
CNN (image only)	0.555	11.32%
Fusion (proposed)	0.981	2.22%

Citation
If you use this code, please cite:

Khalifa, Y., Reda, R., Elsayed, A., Ragab, A.E., Atiea, M.A. (2026).
Multimodal Deep Learning for Effective Property Prediction of
Unidirectional Fiber-Reinforced Composites. Journal Name, Volume, Pages.

License
MIT License

text


---

### FILE 3: `train.py`

Create this file in the root folder. This is a clean, self-contained training script.

```python
"""
Two-phase training script for the multimodal CNN-MLP fusion model.

Phase 1 (warm-up): CNN backbone frozen, train MLP + fusion head.
Phase 2 (fine-tune): All layers unfrozen, end-to-end optimization.

Usage:
    python train.py --data_dir /path/to/data --output_dir /path/to/output
"""

import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from models.fusion_model import build_fusion_model
from models.mlp_branch import build_mlp_branch
from models.physics_loss import physics_consistency_loss, mse_only, physics_penalty_only


def parse_args():
    parser = argparse.ArgumentParser(description="Train fusion model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .npy data files")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save trained model")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase1_epochs", type=int, default=20)
    parser.add_argument("--phase2_epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ── Load data ────────────────────────────────────────────────
    print("Loading data...")
    X_img = np.load(os.path.join(args.data_dir, "X_img_trainpool.npy"))
    X_tab = np.load(os.path.join(args.data_dir, "X_realistic_trainpool.npy"))
    y = np.load(os.path.join(args.data_dir, "y_trainpool.npy"))

    # Train/validation split (80/20, stratified by Vf if available)
    tr_idx, va_idx = train_test_split(
        np.arange(len(X_img)), test_size=0.20,
        random_state=args.seed, shuffle=True
    )

    # Scale tabular inputs and outputs
    sc_tab = StandardScaler().fit(X_tab[tr_idx])
    sc_out = StandardScaler().fit(y[tr_idx])

    X_tab_tr = sc_tab.transform(X_tab[tr_idx]).astype(np.float32)
    X_tab_va = sc_tab.transform(X_tab[va_idx]).astype(np.float32)
    y_tr = sc_out.transform(y[tr_idx]).astype(np.float32)
    y_va = sc_out.transform(y[va_idx]).astype(np.float32)

    # Data augmentation (physically valid for UD composites)
    aug = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=args.seed),
        keras.layers.RandomFlip("vertical", seed=args.seed + 1),
        keras.layers.RandomRotation(0.5, fill_mode="reflect", seed=args.seed + 2),
    ])

    def augment_fn(inputs, label):
        img = aug(tf.expand_dims(inputs["image_input"], 0), training=True)[0]
        return {"image_input": img, "tabular_input": inputs["tabular_input"]}, label

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            ({"image_input": X_img[tr_idx], "tabular_input": X_tab_tr}, y_tr)
        )
        .shuffle(len(tr_idx), seed=args.seed)
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            ({"image_input": X_img[va_idx], "tabular_input": X_tab_va}, y_va)
        )
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Build model ──────────────────────────────────────────────
    custom_objects = {
        "physics_consistency_loss": physics_consistency_loss,
        "mse_only": mse_only,
        "physics_penalty_only": physics_penalty_only,
    }

    model = build_fusion_model(img_size=args.img_size, seed=args.seed)
    print(f"Total parameters: {model.count_params():,}")

    # ── Phase 1: Warm-up (frozen branches) ──────────────────────
    print("\n=== PHASE 1: Warm-up ===")
    for layer in model.layers:
        if layer.name in ("cnn_features", "mlp_branch"):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=physics_consistency_loss,
        metrics=[mse_only, physics_penalty_only],
    )

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.phase1_epochs, verbose=1,
    )

    # ── Phase 2: End-to-end fine-tuning ─────────────────────────
    print("\n=== PHASE 2: Fine-tuning ===")
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=physics_consistency_loss,
        metrics=[mse_only, physics_penalty_only],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=1e-7
        ),
    ]

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.phase2_epochs, verbose=1,
        callbacks=callbacks,
    )

    # ── Save ─────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "fusion_model.keras")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    with open(os.path.join(args.output_dir, "scaler_tab.pkl"), "wb") as f:
        pickle.dump(sc_tab, f)
    with open(os.path.join(args.output_dir, "scaler_output.pkl"), "wb") as f:
        pickle.dump(sc_out, f)
    print("Scalers saved.")


if __name__ == "__main__":
    main()
