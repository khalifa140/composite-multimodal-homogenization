"""
Transverse-isotropy physics-consistency loss for unidirectional composites.

Enforces E22 ≈ E33, G12 ≈ G13, v12 ≈ v13 symmetry constraints
on model predictions, acting as an analytical denoiser that suppresses
finite-RVE numerical artifacts.

Reference: Eq. 5-7 in the associated manuscript.
"""

import tensorflow as tf

# Output index mapping (must match training target order)
IDX = {
    'E11': 0, 'E22': 1, 'E33': 2,
    'G12': 3, 'G13': 4, 'G23': 5,
    'v12': 6, 'v13': 7, 'v23': 8,
}

LAMBDA_PHYSICS = 0.1


def physics_consistency_loss(y_true, y_pred):
    """MSE + lambda * transverse-isotropy symmetry penalty."""
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    diff_E = y_pred[:, IDX['E22']] - y_pred[:, IDX['E33']]
    diff_G = y_pred[:, IDX['G12']] - y_pred[:, IDX['G13']]
    diff_v = y_pred[:, IDX['v12']] - y_pred[:, IDX['v13']]

    physics = tf.reduce_mean(
        tf.square(diff_E) + tf.square(diff_G) + tf.square(diff_v)
    )
    return mse + LAMBDA_PHYSICS * physics


def mse_only(y_true, y_pred):
    """Pure MSE loss (for ablation variants without physics)."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def physics_penalty_only(y_true, y_pred):
    """Symmetry penalty only (for monitoring during training)."""
    diff_E = y_pred[:, IDX['E22']] - y_pred[:, IDX['E33']]
    diff_G = y_pred[:, IDX['G12']] - y_pred[:, IDX['G13']]
    diff_v = y_pred[:, IDX['v12']] - y_pred[:, IDX['v13']]
    return tf.reduce_mean(
        tf.square(diff_E) + tf.square(diff_G) + tf.square(diff_v)
    )
