import numpy as np
import torch
import tensorflow as tf

from utils.reproducibility import load_yaml_config, set_global_seed
from utils.compare_outputs import summarize_output_diff
from models.torch_mlp import TorchMLP
from models.tf_mlp import create_tf_mlp


def align_weights(torch_model: TorchMLP, tf_model: tf.keras.Model):
    tf_weights = tf_model.get_weights()
    state_dict = torch_model.state_dict()
    state_dict["fc1.weight"] = torch.from_numpy(tf_weights[0].T)
    state_dict["fc1.bias"] = torch.from_numpy(tf_weights[1])
    state_dict["fc2.weight"] = torch.from_numpy(tf_weights[2].T)
    state_dict["fc2.bias"] = torch.from_numpy(tf_weights[3])
    torch_model.load_state_dict(state_dict)


def test_forward_outputs_close():
    cfg = load_yaml_config("config/settings.yaml")
    set_global_seed(cfg["seed"])

    inp = cfg["model"]["input_dim"]
    hid = cfg["model"]["hidden_dim"]
    out = cfg["model"]["output_dim"]
    bs = cfg["data"]["batch_size"]

    tf_model = create_tf_mlp(inp, hid, out)
    torch_model = TorchMLP(inp, hid, out)
    align_weights(torch_model, tf_model)

    x_np = np.random.rand(bs, inp).astype(np.float32)
    x_t = torch.from_numpy(x_np)
    x_tf = tf.convert_to_tensor(x_np)

    torch_out = torch_model(x_t).detach().numpy()
    tf_out = tf_model(x_tf).numpy()

    diff = summarize_output_diff(torch_out, tf_out)

    assert diff["mae"] < 1e-5
    assert diff["cosine_similarity"] > 0.9999
