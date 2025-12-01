import os
import numpy as np
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from models.torch_mlp import TorchMLP
from models.tf_mlp import create_tf_mlp
from utils.reproducibility import set_global_seed, load_yaml_config
from utils.sync_weights import sync_tf_to_torch
from utils.losses import torch_mse_loss, tf_mse_loss
from utils.gradients import compute_torch_gradients, compute_tf_gradients
from utils.compare_outputs import summarize_output_diff



def align_gradients_per_layer(torch_grads_list, tf_grads_list):
    """
    Align per-layer gradients between PyTorch and TF.

    For Linear/Dense weights:
      Torch: (out_features, in_features)
      TF:    (in_features, out_features)

    We detect this and transpose the TF gradient so shapes match.

    Returns:
      aligned_torch: list of np.ndarray
      aligned_tf:    list of np.ndarray
    """
    aligned_torch = []
    aligned_tf = []

    for g_torch, g_tf in zip(torch_grads_list, tf_grads_list):
        gt = np.asarray(g_torch)
        gf = np.asarray(g_tf)

        if gt.shape != gf.shape:
            # If one is the transpose of the other, fix it
            if gt.ndim == 2 and gf.shape == gt.T.shape:
                gf = gf.T
            else:
                # Shapes don't match and aren't simple transposes;
                # this shouldn't happen with our current MLPs.
                raise ValueError(
                    f"Gradient shape mismatch that is not a transpose: "
                    f"torch {gt.shape} vs tf {gf.shape}"
                )

        aligned_torch.append(gt)
        aligned_tf.append(gf)

    return aligned_torch, aligned_tf


def main():
    cfg = load_yaml_config("config/settings.yaml")
    set_global_seed(cfg["seed"])

    input_dim = cfg["model"]["input_dim"]
    output_dim = cfg["model"]["output_dim"]
    hidden_dims = cfg["model"].get("hidden_dims", [cfg["model"]["hidden_dim"]])
    activations = cfg["model"].get("activations", ["relu"] * len(hidden_dims))

    # 1. Build models
    tf_model = create_tf_mlp(input_dim, hidden_dims, output_dim, activations)
    torch_model = TorchMLP(input_dim, hidden_dims, output_dim, activations)
    torch_model.eval()

    # 2. Sync TF→Torch weights
    sync_tf_to_torch(tf_model, torch_model)

    # 3. Make synthetic input & target
    batch_size = cfg["data"]["batch_size"]
    x_np = np.random.randn(batch_size, input_dim).astype(np.float32)
    y_np = np.random.randn(batch_size, output_dim).astype(np.float32)

    x_torch = torch.from_numpy(x_np)
    y_torch = torch.from_numpy(y_np)

    x_tf = tf.convert_to_tensor(x_np)
    y_tf = tf.convert_to_tensor(y_np)

    # 4. Compute gradients (raw)
    torch_grads_list, _ = compute_torch_gradients(
        torch_model, x_torch, y_torch, loss_fn=torch_mse_loss
    )
    tf_grads_list, _ = compute_tf_gradients(
        tf_model, x_tf, y_tf, loss_fn=tf_mse_loss
    )

    # 5. Align gradients per layer (fix transpose mismatch for weights)
    aligned_torch_grads, aligned_tf_grads = align_gradients_per_layer(
        torch_grads_list, tf_grads_list
    )

    # 6. Global gradient comparison (after alignment)
    torch_flat = np.concatenate([g.ravel() for g in aligned_torch_grads], axis=0)
    tf_flat = np.concatenate([g.ravel() for g in aligned_tf_grads], axis=0)

    summary = summarize_output_diff(torch_flat, tf_flat)

    print("\n=== Gradient Comparison (Global Vector, Aligned) ===")
    print(f"MAE:               {summary['mae']}")
    print(f"Cosine similarity: {summary['cosine_similarity']}")
    print(f"Cosine distance:   {summary['cosine_distance']}")

    # 7. Layer-wise comparison (after alignment)
    print("\n=== Layer-wise Gradient Differences (Aligned) ===")
    for i, (g_t, g_f) in enumerate(zip(aligned_torch_grads, aligned_tf_grads)):
        s = summarize_output_diff(g_t, g_f)
        print(
            f"Layer {i}: shape={g_t.shape}, "
            f"MAE={s['mae']:.4e}, "
            f"cos_sim={s['cosine_similarity']:.8f}"
        )


if __name__ == "__main__":
    main()
