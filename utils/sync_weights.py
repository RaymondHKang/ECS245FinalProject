import numpy as np
import torch
from torch import nn
from tensorflow import keras

def sync_tf_to_torch(tf_model: keras.Model, torch_model: nn.Module) -> None:
    """
    Copy weights from a Keras MLP into an equivalent PyTorch MLP.

    Assumes:
      - tf_model: built via create_tf_mlp
      - torch_model: built via TorchMLP
      - All Dense <-> Linear layers correspond in order.

    We:
      - Collect all Keras Dense layers (in order)
      - Collect all PyTorch Linear modules (in order)
      - For each pair:
          torch.weight = tf.weight.T
          torch.bias   = tf.bias
    """
    # 1) Get all Dense layers (ignore Input, etc.)
    tf_dense_layers = [
        layer
        for layer in tf_model.layers
        if isinstance(layer, keras.layers.Dense)
    ]

    # 2) Get all Linear layers in Torch (in order of appearance)
    torch_linear_layers = [
        m for m in torch_model.modules() if isinstance(m, nn.Linear)
    ]

    if len(tf_dense_layers) != len(torch_linear_layers):
        raise ValueError(
            f"Mismatch in Dense/Linear layer count: "
            f"TF has {len(tf_dense_layers)}, Torch has {len(torch_linear_layers)}"
        )

    # 3) Copy all weights layer-by-layer
    for tf_layer, torch_layer in zip(tf_dense_layers, torch_linear_layers):
        tf_w, tf_b = tf_layer.get_weights()  # (in_features, out_features), (out_features,)

        # Keras Dense: (in_features, out_features)
        # Torch Linear: (out_features, in_features)
        tf_w_t = tf_w.T

        with torch.no_grad():
            # Ensure dtype match
            w_tensor = torch.from_numpy(tf_w_t).to(torch_layer.weight.dtype)
            b_tensor = torch.from_numpy(tf_b).to(torch_layer.bias.dtype)

            torch_layer.weight.copy_(w_tensor)
            torch_layer.bias.copy_(b_tensor)
