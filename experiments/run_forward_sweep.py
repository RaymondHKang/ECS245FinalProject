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


SEEDS = [0, 1, 2, 42]
BATCH_SIZES = [1, 2, 8, 32]


def run_single_case(cfg, seed: int, batch_size: int):
    """Run one forward comparison for a given (seed, batch_size)."""

    # 1) Seed everything
    set_global_seed(seed)

    # 2) Build models
    input_dim = cfg["model"]["input_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]
    output_dim = cfg["model"]["output_dim"]

    tf_model = create_tf_mlp(input_dim, hidden_dim, output_dim)
    torch_model = TorchMLP(input_dim, hidden_dim, output_dim)
    torch_model.eval()

    # 3) Sync weights TF -> Torch
    sync_tf_to_torch(tf_model, torch_model)

    # 4) Generate random input for this case
    x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_tf = x_np  # Keras accepts NumPy

    with torch.no_grad():
        torch_out = torch_model(x_torch).detach().numpy()
    tf_out = tf_model(x_tf, training=False).numpy()

    # 5) Summarize differences
    summary = summarize_output_diff(torch_out, tf_out)

    return {
        "seed": seed,
        "batch_size": batch_size,
        "mae": float(summary["mae"]),
        "cosine_similarity": float(summary["cosine_similarity"]),
        "cosine_distance": float(summary["cosine_distance"]),
    }


def main():
    cfg = load_yaml_config("config/settings.yaml")

    results = []
    print("=== Forward Sweep: PyTorch vs TensorFlow ===")
    print("seed\tbatch\tMAE\t\tcos_sim\t\tcos_dist")

    for seed in SEEDS:
        for batch_size in BATCH_SIZES:
            res = run_single_case(cfg, seed, batch_size)
            results.append(res)
            print(
                f"{res['seed']}\t{res['batch_size']}\t"
                f"{res['mae']:.8e}\t"
                f"{res['cosine_similarity']:.8f}\t"
                f"{res['cosine_distance']:.8e}"
            )

    # Optional: write to CSV for later analysis
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "forward_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("seed,batch_size,mae,cosine_similarity,cosine_distance\n")
        for r in results:
            f.write(
                f"{r['seed']},{r['batch_size']},"
                f"{r['mae']},{r['cosine_similarity']},{r['cosine_distance']}\n"
            )

    print(f"\nSaved detailed sweep results to: {csv_path}")


if __name__ == "__main__":
    main()
