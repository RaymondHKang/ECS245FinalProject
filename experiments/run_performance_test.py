import os
import numpy as np
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from utils.reproducibility import load_yaml_config, set_global_seed
from utils.measure_perf import time_forward_pass, measure_memory_for_call
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


def main():
    cfg = load_yaml_config("config/settings.yaml")
    set_global_seed(cfg["seed"])

    m = cfg["model"]
    d = cfg["data"]
    perf = cfg["performance"]

    input_dim = m["input_dim"]
    hidden_dim = m["hidden_dim"]
    output_dim = m["output_dim"]
    batch_size = d["batch_size"]
    num_runs = perf["num_runs"]

    tf_model = create_tf_mlp(input_dim, hidden_dim, output_dim)
    torch_model = TorchMLP(input_dim, hidden_dim, output_dim)
    align_weights(torch_model, tf_model)

    x_np = np.random.rand(batch_size, input_dim).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_tf = tf.convert_to_tensor(x_np)

    def torch_forward():
        _ = torch_model(x_torch)

    def tf_forward():
        _ = tf_model(x_tf, training=False)

    print("=== Performance Benchmark ===")
    # Time
    t_avg, t_std = time_forward_pass(torch_forward, num_runs=num_runs)
    f_avg, f_std = time_forward_pass(tf_forward, num_runs=num_runs)

    print(f"PyTorch forward:  avg={t_avg*1000:.3f} ms, std={t_std*1000:.3f} ms")
    print(f"TensorFlow fwd:   avg={f_avg*1000:.3f} ms, std={f_std*1000:.3f} ms")

    # Memory
    t_mem = measure_memory_for_call(torch_forward)
    f_mem = measure_memory_for_call(tf_forward)

    print(f"Approx PyTorch forward memory delta:   {t_mem:.3f} MB")
    print(f"Approx TensorFlow forward memory delta:{f_mem:.3f} MB")


if __name__ == "__main__":
    main()
