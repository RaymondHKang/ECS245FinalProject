import numpy as np
import torch
from torch import nn
from tensorflow import keras

def sync_tf_to_torch(tf_model: keras.Model, torch_model: nn.Module) -> None:
    """
    Copy weights from a simple Keras MLP into the equivalent PyTorch MLP.

    Assumes architecture:
      TF: Input -> Dense(hidden, relu) -> Dense(output)
      Torch: Linear(input, hidden) -> ReLU -> Linear(hidden, output)
    """

    # Grab TF layers (ignore Input layer)
    tf_dense_layers = [layer for layer in tf_model.layers if isinstance(layer, keras.layers.Dense)]
    if len(tf_dense_layers) != 2:
        raise ValueError(f"Expected exactly 2 Dense layers, got {len(tf_dense_layers)}")

    # Unpack TF params
    tf_w1, tf_b1 = tf_dense_layers[0].get_weights()
    tf_w2, tf_b2 = tf_dense_layers[1].get_weights()

    # Grab Torch layers
    torch_layers = [m for m in torch_model.modules() if isinstance(m, nn.Linear)]
    if len(torch_layers) != 2:
        raise ValueError(f"Expected exactly 2 Linear layers, got {len(torch_layers)}")

    lin1, lin2 = torch_layers

    # Keras Dense: (in_features, out_features)
    # PyTorch Linear: (out_features, in_features)
    with torch.no_grad():
        lin1.weight.copy_(torch.from_numpy(tf_w1.T))  # transpose
        lin1.bias.copy_(torch.from_numpy(tf_b1))

        lin2.weight.copy_(torch.from_numpy(tf_w2.T))
        lin2.bias.copy_(torch.from_numpy(tf_b2))
