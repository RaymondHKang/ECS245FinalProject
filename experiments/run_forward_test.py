import os
import numpy as np
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from utils.reproducibility import load_yaml_config, set_global_seed
from utils.compare_outputs import summarize_output_diff
from models.torch_mlp import TorchMLP
from models.tf_mlp import create_tf_mlp
from utils.sync_weights import sync_tf_to_torch

def main():
    cfg = load_yaml_config("config/settings.yaml")
    set_global_seed(cfg["seed"])

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    weights_cfg = cfg["weights"]

    input_dim = model_cfg["input_dim"]
    hidden_dim = model_cfg["hidden_dim"]
    output_dim = model_cfg["output_dim"]
    batch_size = data_cfg["batch_size"]

    tf_model = create_tf_mlp(input_dim, hidden_dim, output_dim)
    torch_model = TorchMLP(input_dim, hidden_dim, output_dim)

    sync_tf_to_torch(tf_model, torch_model)

    # synthetic input (also save to file once)
    os.makedirs(os.path.dirname(data_cfg["synthetic_inputs_path"]), exist_ok=True)
    if not os.path.exists(data_cfg["synthetic_inputs_path"]):
        x_np = np.random.rand(batch_size, input_dim).astype(np.float32)
        np.save(data_cfg["synthetic_inputs_path"], x_np)
    else:
        x_np = np.load(data_cfg["synthetic_inputs_path"])

    x_torch = torch.from_numpy(x_np)
    x_tf = tf.convert_to_tensor(x_np)

    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(x_torch).detach().numpy()
    tf_out = tf_model(x_tf, training=False).numpy()

    summary = summarize_output_diff(torch_out, tf_out)

    print("=== Forward Output Comparison (PyTorch vs TensorFlow) ===")
    print(f"MAE:               {summary['mae']:.8f}")
    print(f"Cosine similarity: {summary['cosine_similarity']:.8f}")
    print(f"Cosine distance:   {summary['cosine_distance']:.8f}")


if __name__ == "__main__":
    main()
