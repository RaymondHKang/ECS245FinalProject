import numpy as np
import torch
import tensorflow as tf

from utils.reproducibility import load_yaml_config, set_global_seed
from models.torch_mlp import TorchMLP
from models.tf_mlp import create_tf_mlp


def align_weights(torch_model: TorchMLP, tf_model: tf.keras.Model):
    """Copy TF weights into the PyTorch model (2-layer MLP assumption)."""
    tf_weights = tf_model.get_weights()
    state_dict = torch_model.state_dict()
    state_dict["fc1.weight"] = torch.from_numpy(tf_weights[0].T)
    state_dict["fc1.bias"] = torch.from_numpy(tf_weights[1])
    state_dict["fc2.weight"] = torch.from_numpy(tf_weights[2].T)
    state_dict["fc2.bias"] = torch.from_numpy(tf_weights[3])
    torch_model.load_state_dict(state_dict)


def test_weight_alignment_basic():
    cfg = load_yaml_config("config/settings.yaml")
    set_global_seed(cfg["seed"])

    inp = cfg["model"]["input_dim"]
    hid = cfg["model"]["hidden_dim"]
    out = cfg["model"]["output_dim"]

    tf_model = create_tf_mlp(inp, hid, out)
    torch_model = TorchMLP(inp, hid, out)

    # after alignment, one random input should give almost identical outputs
    align_weights(torch_model, tf_model)

    x_np = np.random.rand(5, inp).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_tf = tf.convert_to_tensor(x_np)

    torch_out = torch_model(x_torch).detach().numpy()
    tf_out = tf_model(x_tf).numpy()

    assert torch_out.shape == tf_out.shape
    assert np.allclose(torch_out, tf_out, atol=1e-5)
