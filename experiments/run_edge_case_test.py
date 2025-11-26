import numpy as np
import torch
import tensorflow as tf

from utils.reproducibility import load_yaml_config, set_global_seed
from utils.edge_case_inputs import generate_edge_case_values, generate_edge_case_batch
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

    input_dim = m["input_dim"]
    hidden_dim = m["hidden_dim"]
    output_dim = m["output_dim"]
    batch_size = d["batch_size"]

    tf_model = create_tf_mlp(input_dim, hidden_dim, output_dim)
    torch_model = TorchMLP(input_dim, hidden_dim, output_dim)
    align_weights(torch_model, tf_model)

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
