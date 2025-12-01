import os
import numpy as np
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras

from models.torch_mlp import TorchMLP
from models.tf_mlp import create_tf_mlp
from utils.reproducibility import set_global_seed, load_yaml_config
from utils.sync_weights import sync_tf_to_torch
from utils.losses import torch_mse_loss, tf_mse_loss
from utils.gradients import compute_torch_gradients, compute_tf_gradients
from utils.compare_outputs import summarize_output_diff


DTYPES = ["float32", "float64"]
SEEDS = [0, 1, 2]
BATCH_SIZES = [1, 8, 32]

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


def align_gradients_per_layer(torch_grads_list, tf_grads_list):
    """
    Align per-layer gradients between PyTorch and TF.

    For Linear/Dense weights:
      Torch: (out_features, in_features)
      TF:    (in_features, out_features)

    We detect this and transpose the TF gradient so shapes match.
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
                raise ValueError(
                    f"Gradient shape mismatch that is not a transpose: "
                    f"torch {gt.shape} vs tf {gf.shape}"
                )

        aligned_torch.append(gt)
        aligned_tf.append(gf)

    return aligned_torch, aligned_tf


def build_models(cfg, arch_cfg, dtype_name):
    input_dim = cfg["model"]["input_dim"]
    output_dim = cfg["model"]["output_dim"]
    hidden_dims = arch_cfg["hidden_dims"]
    activations = arch_cfg["activations"]

    set_tf_floatx(dtype_name)
    if dtype_name == "float32":
        tf_dtype = tf.float32
        torch_dtype = torch.float32
    elif dtype_name == "float64":
        tf_dtype = tf.float64
        torch_dtype = torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    tf_model = create_tf_mlp(
        input_dim=input_dim,
        hidden_dim=hidden_dims,
        output_dim=output_dim,
        activations=activations,
    )

    torch_model = TorchMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dims,
        output_dim=output_dim,
        activations=activations,
    )
    torch_model = torch_model.to(torch_dtype)
    torch_model.eval()

    sync_tf_to_torch(tf_model, torch_model)

    return tf_model, torch_model, tf_dtype, torch_dtype


def run_single_case(cfg, arch_cfg, dtype_name, seed, batch_size):
    set_global_seed(seed)

    tf_model, torch_model, tf_dtype, torch_dtype = build_models(cfg, arch_cfg, dtype_name)

    input_dim = cfg["model"]["input_dim"]
    output_dim = cfg["model"]["output_dim"]

    # Inputs and targets
    x_np = np.random.randn(batch_size, input_dim)
    y_np = np.random.randn(batch_size, output_dim)

    if dtype_name == "float32":
        x_np = x_np.astype(np.float32)
        y_np = y_np.astype(np.float32)
    else:
        x_np = x_np.astype(np.float64)
        y_np = y_np.astype(np.float64)

    x_torch = torch.from_numpy(x_np).to(torch_dtype)
    y_torch = torch.from_numpy(y_np).to(torch_dtype)

    x_tf = tf.convert_to_tensor(x_np, dtype=tf_dtype)
    y_tf = tf.convert_to_tensor(y_np, dtype=tf_dtype)

    # Compute gradients (raw)
    torch_grads_list, _ = compute_torch_gradients(
        torch_model, x_torch, y_torch, loss_fn=torch_mse_loss
    )
    tf_grads_list, _ = compute_tf_gradients(
        tf_model, x_tf, y_tf, loss_fn=tf_mse_loss
    )

    # Align per-layer gradients
    aligned_torch_grads, aligned_tf_grads = align_gradients_per_layer(
        torch_grads_list, tf_grads_list
    )

    # Global flattened vectors
    torch_flat = np.concatenate([g.ravel() for g in aligned_torch_grads], axis=0)
    tf_flat = np.concatenate([g.ravel() for g in aligned_tf_grads], axis=0)

    # Global metrics
    grad_summary = summarize_output_diff(torch_flat, tf_flat)

    # Gradient norms
    torch_norm = float(np.linalg.norm(torch_flat))
    tf_norm = float(np.linalg.norm(tf_flat))

    # Layer-wise metrics (summarized)
    layer_maes = []
    layer_cos = []
    for g_t, g_f in zip(aligned_torch_grads, aligned_tf_grads):
        s = summarize_output_diff(g_t, g_f)
        layer_maes.append(s["mae"])
        layer_cos.append(s["cosine_similarity"])

    max_layer_mae = float(max(layer_maes))
    min_layer_cos = float(min(layer_cos))

    # NaN / Inf checks
    any_nan_inf_torch = bool(
        np.isnan(torch_flat).any() or np.isinf(torch_flat).any()
    )
    any_nan_inf_tf = bool(
        np.isnan(tf_flat).any() or np.isinf(tf_flat).any()
    )

    return {
        "arch": arch_cfg["name"],
        "dtype": dtype_name,
        "seed": seed,
        "batch_size": batch_size,
        "grad_mae": float(grad_summary["mae"]),
        "grad_cos_sim": float(grad_summary["cosine_similarity"]),
        "torch_grad_norm": torch_norm,
        "tf_grad_norm": tf_norm,
        "max_layer_mae": max_layer_mae,
        "min_layer_cos": min_layer_cos,
        "nan_inf_torch": any_nan_inf_torch,
        "nan_inf_tf": any_nan_inf_tf,
    }


def main():
    cfg = load_yaml_config("config/settings.yaml")

    results = []
    print("=== Gradient Sweep: arch × activation × dtype × seed × batch ===")
    print(
        "arch\t\tdtype\tseed\tbatch\t"
        "grad_MAE\t\tgrad_cos_sim\t"
        "max_layer_MAE\tmin_layer_cos"
    )

    for arch_cfg in ARCH_CONFIGS:
        for dtype_name in DTYPES:
            for seed in SEEDS:
                for batch_size in BATCH_SIZES:
                    res = run_single_case(cfg, arch_cfg, dtype_name, seed, batch_size)
                    results.append(res)

                    print(
                        f"{res['arch']}\t{res['dtype']}\t{res['seed']}\t{res['batch_size']}\t"
                        f"{res['grad_mae']:.8e}\t"
                        f"{res['grad_cos_sim']:.8f}\t"
                        f"{res['max_layer_mae']:.8e}\t"
                        f"{res['min_layer_cos']:.8f}"
                    )

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "gradient_sweep.csv")
    with open(csv_path, "w") as f:
        f.write(
            "arch,dtype,seed,batch_size,"
            "grad_mae,grad_cos_sim,torch_grad_norm,tf_grad_norm,"
            "max_layer_mae,min_layer_cos,nan_inf_torch,nan_inf_tf\n"
        )
        for r in results:
            f.write(
                f"{r['arch']},{r['dtype']},{r['seed']},{r['batch_size']},"
                f"{r['grad_mae']},{r['grad_cos_sim']},"
                f"{r['torch_grad_norm']},{r['tf_grad_norm']},"
                f"{r['max_layer_mae']},{r['min_layer_cos']},"
                f"{r['nan_inf_torch']},{r['nan_inf_tf']}\n"
            )

    max_mae = max(r["grad_mae"] for r in results)
    min_cos = min(r["grad_cos_sim"] for r in results)
    print(f"\nMax global grad MAE across all cases: {max_mae}")
    print(f"Min global grad cosine similarity:    {min_cos}")
    print(f"Saved detailed gradient sweep results to: {csv_path}")


if __name__ == "__main__":
    main()
