import os
import numpy as np
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from utils.reproducibility import load_yaml_config, set_global_seed
from utils.edge_case_inputs import generate_edge_case_values, generate_edge_case_batch
from models.torch_mlp import TorchMLP
from models.tf_mlp import create_tf_mlp
from utils.sync_weights import sync_tf_to_torch

def main():
    cfg = load_yaml_config("config/settings.yaml")
    set_global_seed(cfg["seed"])

    m = cfg["model"]
    d = cfg["data"]

    input_dim = m["input_dim"]
    hidden_dim = m["hidden_dim"]
    output_dim = m["output_dim"]
    batch_size = d["batch_size"]

    tf_model = create_tf_mlp(input_dim, hidden_dim, output_dim)
    torch_model = TorchMLP(input_dim, hidden_dim, output_dim)
    sync_tf_to_torch(tf_model, torch_model)

    values = generate_edge_case_values()

    print("=== Edge-case Input Stability Test ===")
    for v in values:
        x_np = generate_edge_case_batch(v, batch_size, input_dim)
        x_t = torch.from_numpy(x_np)
        x_tf = tf.convert_to_tensor(x_np)

        try:
            torch_out = torch_model(x_t).detach().numpy()
            torch_status = "ok"
        except Exception as e:
            torch_out = None
            torch_status = f"error: {e}"

        try:
            tf_out = tf_model(x_tf).numpy()
            tf_status = "ok"
        except Exception as e:
            tf_out = None
            tf_status = f"error: {e}"

        print(f"\nInput value: {v}")
        print(f"  PyTorch status:   {torch_status}")
        print(f"  TensorFlow status:{tf_status}")

        if torch_out is not None and tf_out is not None:
            # Print simple statistics to avoid flooding output
            print(f"  PyTorch output stats: mean={np.nanmean(torch_out):.6f}, "
                  f"min={np.nanmin(torch_out):.6f}, max={np.nanmax(torch_out):.6f}")
            print(f"  TF output stats:      mean={np.nanmean(tf_out):.6f}, "
                  f"min={np.nanmin(tf_out):.6f}, max={np.nanmax(tf_out):.6f}")


if __name__ == "__main__":
    main()
