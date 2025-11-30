# utils/losses.py

import torch
import tensorflow as tf
import numpy as np


def torch_mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def tf_mse_loss(target, pred):
    """
    MSE loss for TF/Keras with safe dtype handling.

    We explicitly cast pred to target's dtype so that
    pred - target is always well-defined (float32/float64).
    """
    # Ensure both are the same dtype
    pred = tf.cast(pred, target.dtype)
    return tf.reduce_mean(tf.square(pred - target))
