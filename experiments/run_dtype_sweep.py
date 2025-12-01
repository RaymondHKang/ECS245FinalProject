# experiments/run_dtype_sweep.py

import os
import numpy as np
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras

from models.tf_mlp import create_tf_mlp
from models.torch_mlp import TorchMLP
from utils.reproducibility import load_yaml_config, set_global_seed
from utils.sync_weights import sync_tf_to_torch
from utils.compare_outputs import summarize_output_diff


DTYPES = ["float32", "float64"]
SEEDS = [0, 1, 2, 42]
BATCH_SIZES = [1, 2, 8, 32]


def build_models_in_dtype(cfg, dtype_name: str):
    input_dim = cfg["model"]["input_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]
    output_dim = cfg["model"]["output_dim"]

    if dtype_name == "float32":
        torch_dtype = torch.float32
        tf_dtype = tf.float32
        keras.backend.set_floatx("float32")
    elif dtype_name == "float64":
        torch_dtype = torch.float64
        tf_dtype = tf.float64
        keras.backend.set_floatx("float64")
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    # Build TF model in the chosen floatx
    tf_model = create_tf_mlp(input_dim, hidden_dim, output_dim)
    # Ensure its variables are in the right dtype
    tf_model = tf_model.astype(tf_dtype) if hasattr(tf_model, "astype") else tf_model

    # Build Torch model and cast parameters
    torch_model = TorchMLP(input_dim, hidden_dim, output_dim)
    torch_model = torch_model.to(torch_dtype)
    torch_model.eval()

    # Sync weights TF -> Torch (note: sync uses numpy, which is dtype-agnostic;
    # but underlying tensors/arrays will be in the right dtype)
    sync_tf_to_torch(tf_model, torch_model)

    return tf_model, torch_model, tf_dtype, torch_dtype


def run_single_case(cfg, dtype_name: str, seed: int, batch_size: int):
    set_global_seed(seed)

    tf_model, torch_model, tf_dtype, torch_dtype = build_models_in_dtype(cfg, dtype_name)

    input_dim = cfg["model"]["input_dim"]
    x_np = np.random.randn(batch_size, input_dim)

    if dtype_name == "float32":
        x_np = x_np.astype(np.float32)
    elif dtype_name == "float64":
        x_np = x_np.astype(np.float64)

    x_torch = torch.from_numpy(x_np).to(torch_dtype)
    x_tf = tf.convert_to_tensor(x_np, dtype=tf_dtype)

    with torch.no_grad():
        torch_out = torch_model(x_torch).cpu().numpy()
    tf_out = tf_model(x_tf, training=False).numpy()

    summary = summarize_output_diff(torch_out, tf_out)

    return {
        "dtype": dtype_name,
        "seed": seed,
        "batch_size": batch_size,
        "mae": float(summary["mae"]),
        "cosine_similarity": float(summary["cosine_similarity"]),
        "cosine_distance": float(summary["cosine_distance"]),
    }


def main():
    cfg = load_yaml_config("config/settings.yaml")

    results = []
    print("=== Dtype Sweep: PyTorch vs TensorFlow ===")
    print("dtype\tseed\tbatch\tMAE\t\tcos_sim\t\tcos_dist")

    for dtype_name in DTYPES:
        for seed in SEEDS:
            for batch_size in BATCH_SIZES:
                res = run_single_case(cfg, dtype_name, seed, batch_size)
                results.append(res)
                print(
                    f"{res['dtype']}\t{res['seed']}\t{res['batch_size']}\t"
                    f"{res['mae']:.8e}\t"
                    f"{res['cosine_similarity']:.8f}\t"
                    f"{res['cosine_distance']:.8e}"
                )

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "dtype_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("dtype,seed,batch_size,mae,cosine_similarity,cosine_distance\n")
        for r in results:
            f.write(
                f"{r['dtype']},{r['seed']},{r['batch_size']},"
                f"{r['mae']},{r['cosine_similarity']},{r['cosine_distance']}\n"
            )

    max_mae = max(r["mae"] for r in results)
    min_cos = min(r["cosine_similarity"] for r in results)
    print(f"\nMax MAE across all cases: {max_mae}")
    print(f"Min cosine similarity across all cases: {min_cos}")
    print(f"Saved detailed dtype sweep results to: {csv_path}")


if __name__ == "__main__":
    main()
