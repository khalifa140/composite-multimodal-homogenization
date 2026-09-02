"""
Lightweight 3-stage Micro-ResNet backbone for 384x384 RVE images.

Architecture:
  Input 384x384x1
  → Stem (7x7 conv, stride 2 + maxpool) → 96x96x16
  → Stage 1 (2 residual blocks, 32 filters) → 96x96x32
  → Stage 2 (2 residual blocks, 64 filters, stride 2) → 48x48x64
  → Stage 3 (2 residual blocks, 128 filters, stride 2) → 24x24x128
  → Global Average Pooling → 128-dim feature vector

Total parameters: ~720K (standalone CNN with dense head)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def _residual_block(x, filters, stride=1, name_prefix='rb'):
    """Single residual block with two 3x3 convolutions."""
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      name=f'{name_prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)

    x = layers.Conv2D(filters, 3, strides=1, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      name=f'{name_prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)

    # Projection shortcut if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 use_bias=False, kernel_initializer='he_normal',
                                 name=f'{name_prefix}_proj')(shortcut)
        shortcut = layers.BatchNormalization(
            name=f'{name_prefix}_proj_bn')(shortcut)

    x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)
    return x


def build_micro_resnet(img_size=384, n_outputs=9,
                       dropout=0.50, l2_reg=1e-3, seed=42,
                       include_head=True):
    """
    Build the Micro-ResNet model.

    Args:
        img_size: Input image dimension (square).
        n_outputs: Number of regression targets.
        dropout: Dropout rate for dense head layers.
        l2_reg: L2 regularization weight for dense layers.
        seed: Random seed for reproducibility.
        include_head: If True, include dense regression head.
                      If False, output is the 128-dim GAP feature vector
                      (used when building the fusion model).

    Returns:
        keras.Model
    """
    tf.random.set_seed(seed)
    reg = regularizers.l2(l2_reg)

    inp = keras.Input(shape=(img_size, img_size, 1), name='image_input')

    # Stem: 384 → 96
    x = layers.Conv2D(16, 7, strides=2, padding='same', use_bias=False,
                      kernel_initializer='he_normal', name='stem_conv')(inp)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('relu', name='stem_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same',
                            name='stem_pool')(x)

    # Stage 1: 96x96x32
    x = _residual_block(x, 32, stride=1, name_prefix='s1b1')
    x = _residual_block(x, 32, stride=1, name_prefix='s1b2')

    # Stage 2: 48x48x64
    x = _residual_block(x, 64, stride=2, name_prefix='s2b1')
    x = _residual_block(x, 64, stride=1, name_prefix='s2b2')

    # Stage 3: 24x24x128
    x = _residual_block(x, 128, stride=2, name_prefix='s3b1')
    x = _residual_block(x, 128, stride=1, name_prefix='s3b2')

    # Global Average Pooling → 128-dim vector
    x = layers.GlobalAveragePooling2D(name='gap')(x)

    if include_head:
        x = layers.Dense(128, activation='relu', kernel_regularizer=reg,
                         kernel_initializer='he_normal', name='head_128')(x)
        x = layers.Dropout(dropout, seed=seed, name='drop_128')(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=reg,
                         kernel_initializer='he_normal', name='head_64')(x)
        x = layers.Dropout(dropout * 0.8, seed=seed + 1, name='drop_64')(x)
        out = layers.Dense(n_outputs, activation='linear', name='output')(x)
        return keras.Model(inputs=inp, outputs=out, name='MicroResNet')
    else:
        return keras.Model(inputs=inp, outputs=x, name='MicroResNet_features')
