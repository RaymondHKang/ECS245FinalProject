import tensorflow as tf
from tensorflow.keras import layers, models


def create_tf_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> tf.keras.Model:
    """
    Simple 2-layer MLP in TensorFlow/Keras with the same architecture as TorchMLP.
    """
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    outputs = layers.Dense(output_dim)(x)  # linear output
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
