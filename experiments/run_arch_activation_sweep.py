import os
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras

from models.tf_mlp import create_tf_mlp
from models.torch_mlp import TorchMLP
from utils.reproducibility import load_yaml_config, set_global_seed
from utils.sync_weights import sync_tf_to_torch
from utils.compare_outputs import summarize_output_diff

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DTYPES = ["float32", "float64"]
SEEDS = [0, 1, 2]
BATCH_SIZES = [1, 8, 32]

# A few reasonably varied architectures/activations
ARCH_CONFIGS = [
    {
        "name": "1x64_relu",
        "hidden_dims": [64],
        "activations": ["relu"],
    },
    {
        "name": "2x64_relu",
        "hidden_dims": [64, 64],
        "activations": ["relu", "relu"],
    },
    {
        "name": "128_64_tanh_relu",
        "hidden_dims": [128, 64],
        "activations": ["tanh", "relu"],
    },
    {
        "name": "3x32_gelu_relu_tanh",
        "hidden_dims": [32, 32, 32],
        "activations": ["gelu", "relu", "tanh"],
    },
    {
        "name": "2x256_sigmoid",
        "hidden_dims": [256, 256],
        "activations": ["sigmoid", "sigmoid"],
    },
    {
        "name": "2x64_elu",
        "hidden_dims": [64, 64],
        "activations": ["elu", "elu"],
    },
]


def set_tf_floatx(dtype_name: str):
    if dtype_name == "float32":
        keras.backend.set_floatx("float32")
    elif dtype_name == "float64":
        keras.backend.set_floatx("float64")
    else:
        raise ValueError(f"Unsupported dtype_name: {dtype_name}")


def build_models(cfg, arch_cfg, dtype_name):
    input_dim = cfg["model"]["input_dim"]
    output_dim = cfg["model"]["output_dim"]
    hidden_dims = arch_cfg["hidden_dims"]
    activations = arch_cfg["activations"]

    # Configure TF dtype
    set_tf_floatx(dtype_name)
    if dtype_name == "float32":
        tf_dtype = tf.float32
        torch_dtype = torch.float32
    else:
        tf_dtype = tf.float64
        torch_dtype = torch.float64

    # Build TF model
    tf_model = create_tf_mlp(
        input_dim=input_dim,
        hidden_dim=hidden_dims,
        output_dim=output_dim,
        activations=activations,
    )
    # Ensure variables are correct dtype
    tf_model = tf_model

    # Build Torch model
    torch_model = TorchMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dims,
        output_dim=output_dim,
        activations=activations,
    )
    torch_model = torch_model.to(torch_dtype)
    torch_model.eval()

    # Sync weights TF -> Torch
    sync_tf_to_torch(tf_model, torch_model)

    return tf_model, torch_model, tf_dtype, torch_dtype


def run_single_case(cfg, arch_cfg, dtype_name, seed, batch_size):
    set_global_seed(seed)

    tf_model, torch_model, tf_dtype, torch_dtype = build_models(cfg, arch_cfg, dtype_name)
    input_dim = cfg["model"]["input_dim"]

    x_np = np.random.randn(batch_size, input_dim)
    if dtype_name == "float32":
        x_np = x_np.astype(np.float32)
    else:
        x_np = x_np.astype(np.float64)

    x_torch = torch.from_numpy(x_np).to(torch_dtype)
    x_tf = tf.convert_to_tensor(x_np, dtype=tf_dtype)

    with torch.no_grad():
        torch_out = torch_model(x_torch).cpu().numpy()
    tf_out = tf_model(x_tf, training=False).numpy()

    summary = summarize_output_diff(torch_out, tf_out)

    return {
        "arch": arch_cfg["name"],
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
    print("=== Arch × Activation × Dtype Sweep: PyTorch vs TensorFlow ===")
    print("arch\t\tdtype\tseed\tbatch\tMAE\t\tcos_sim\t\tcos_dist")

    for arch_cfg in ARCH_CONFIGS:
        for dtype_name in DTYPES:
            for seed in SEEDS:
                for batch_size in BATCH_SIZES:
                    res = run_single_case(cfg, arch_cfg, dtype_name, seed, batch_size)
                    results.append(res)
                    print(
                        f"{res['arch']}\t{res['dtype']}\t{res['seed']}\t{res['batch_size']}\t"
                        f"{res['mae']:.8e}\t"
                        f"{res['cosine_similarity']:.8f}\t"
                        f"{res['cosine_distance']:.8e}"
                    )

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "arch_activation_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("arch,dtype,seed,batch_size,mae,cosine_similarity,cosine_distance\n")
        for r in results:
            f.write(
                f"{r['arch']},{r['dtype']},{r['seed']},{r['batch_size']},"
                f"{r['mae']},{r['cosine_similarity']},{r['cosine_distance']}\n"
            )

    max_mae = max(r["mae"] for r in results)
    min_cos = min(r["cosine_similarity"] for r in results)
    print(f"\nMax MAE across all cases: {max_mae}")
    print(f"Min cosine similarity across all cases: {min_cos}")
    print(f"Saved detailed arch/activation sweep results to: {csv_path}")


if __name__ == "__main__":
    main()
