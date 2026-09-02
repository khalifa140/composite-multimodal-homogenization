"""
MLP branch for processing 8 constituent material properties.

Architecture:
  Input (8,) → Dense(128) → BN → ReLU → Drop(0.25)
             → Dense(64)  → BN → ReLU → Drop(0.20)
             → Dense(32, ReLU)  [material identity vector]

Total parameters: ~13K
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_mlp_branch(n_inputs=8, output_dim=32,
                     l2_reg=1e-4, seed=42):
    """
    Build the tabular MLP branch.

    Args:
        n_inputs: Number of constituent properties (default 8).
        output_dim: Dimension of the material identity vector.
        l2_reg: L2 regularization weight.
        seed: Random seed.

    Returns:
        keras.Model with shape (n_inputs,) → (output_dim,)
    """
    tf.random.set_seed(seed)
    reg = regularizers.l2(l2_reg)

    inp = keras.Input(shape=(n_inputs,), name='tabular_input')

    x = layers.Dense(128, kernel_regularizer=reg,
                     kernel_initializer='glorot_uniform',
                     name='mlp_dense_128')(inp)
    x = layers.BatchNormalization(name='mlp_bn_128')(x)
    x = layers.Activation('relu', name='mlp_relu_128')(x)
    x = layers.Dropout(0.25, seed=seed, name='mlp_drop_128')(x)

    x = layers.Dense(64, kernel_regularizer=reg,
                     kernel_initializer='glorot_uniform',
                     name='mlp_dense_64')(x)
    x = layers.BatchNormalization(name='mlp_bn_64')(x)
    x = layers.Activation('relu', name='mlp_relu_64')(x)
    x = layers.Dropout(0.20, seed=seed + 1, name='mlp_drop_64')(x)

    out = layers.Dense(output_dim, activation='relu',
                       kernel_regularizer=reg,
                       kernel_initializer='glorot_uniform',
                       name='mlp_features')(x)

    return keras.Model(inputs=inp, outputs=out, name='MLP_branch')
