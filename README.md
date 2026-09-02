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
