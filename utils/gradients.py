import numpy as np
import torch
import tensorflow as tf


# -------------------------------------------------------------
#  Helper: flatten all gradient tensors into a single 1D vector
# -------------------------------------------------------------
def _flatten_and_concat(gradient_list):
    flat = [g.reshape(-1) for g in gradient_list]
    return np.concatenate(flat, axis=0)


# -------------------------------------------------------------
#  PyTorch gradient extraction
# -------------------------------------------------------------
def compute_torch_gradients(model, x, y, loss_fn):
    """
    Compute gradients for a PyTorch model.
      model: Torch model
      x: torch.Tensor (batch_size, input_dim)
      y: torch.Tensor targets
      loss_fn: callable producing scalar loss

    Returns:
      gradients_np_list: list of numpy arrays, one per parameter
      gradients_flat:    flattened 1D numpy vector
    """
    model.zero_grad()

    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()

    gradients_list = []
    for p in model.parameters():
        if p.grad is None:
            gradients_list.append(np.zeros_like(p.detach().cpu().numpy()))
        else:
            gradients_list.append(p.grad.detach().cpu().numpy())

    return gradients_list, _flatten_and_concat(gradients_list)


# -------------------------------------------------------------
#  TensorFlow gradient extraction
# -------------------------------------------------------------
def compute_tf_gradients(model, x, y, loss_fn):
    """
    Compute gradients for a TF/Keras model.
      model: Keras model
      x: tensorflow.Tensor
      y: tensorflow.Tensor
      loss_fn: callable producing scalar loss

    Returns:
      gradients_np_list: list of numpy arrays, one per parameter
      gradients_flat:    flattened 1D numpy vector
    """
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)

    gradients = tape.gradient(loss, model.trainable_variables)

    gradients_list = []
    for g, var in zip(gradients, model.trainable_variables):
        if g is None:
            gradients_list.append(np.zeros_like(var.numpy()))
        else:
            gradients_list.append(g.numpy())

    return gradients_list, _flatten_and_concat(gradients_list)
