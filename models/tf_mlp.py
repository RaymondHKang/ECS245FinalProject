import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Sequence, Union, List, Optional


def create_tf_mlp(
    input_dim: int,
    hidden_dim: Union[int, Sequence[int]],
    output_dim: int,
    activations: Optional[Sequence[str]] = None,
) -> tf.keras.Model:
    """
    Flexible MLP in TensorFlow/Keras.

    - hidden_dim can be int (single layer) or Sequence[int] (multiple layers)
    - activations: same length as hidden_dims, from:
        'relu', 'tanh', 'sigmoid', 'gelu', 'elu'
    """
    # Normalize hidden_dim to list
    if isinstance(hidden_dim, int):
        hidden_dims: List[int] = [hidden_dim]
    else:
        hidden_dims = list(hidden_dim)

    if activations is None:
        activations = ["relu"] * len(hidden_dims)
    else:
        activations = list(activations)

    if len(activations) != len(hidden_dims):
        raise ValueError(
            f"Number of activations ({len(activations)}) "
            f"must match number of hidden layers ({len(hidden_dims)})"
        )

    inputs = layers.Input(shape=(input_dim,), dtype=tf.keras.backend.floatx())
    x = inputs

    for h, act in zip(hidden_dims, activations):
        # Keras supports these as string activations
        x = layers.Dense(h, activation=act)(x)

    outputs = layers.Dense(output_dim, activation=None)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
