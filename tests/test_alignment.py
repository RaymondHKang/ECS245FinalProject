import numpy as np
import torch
import tensorflow as tf

from models.tf_mlp import create_tf_mlp
from models.torch_mlp import TorchMLP
from utils.sync_weights import sync_tf_to_torch

SEEDS = [0, 1, 42]
BATCH_SIZES = [1, 4, 16]

def test_weight_alignment_forward_equivalence():
    input_dim = 32
    hidden_dim = 64
    output_dim = 10

    for seed in SEEDS:
        # Set seeds for reproducibility inside the test
        np.random.seed(seed)
        tf.random.set_seed(seed)
        torch.manual_seed(seed)

        # Create models
        tf_model = create_tf_mlp(input_dim, hidden_dim, output_dim)
        torch_model = TorchMLP(input_dim, hidden_dim, output_dim)
        torch_model.eval()

        # Sync weights TF -> Torch
        sync_tf_to_torch(tf_model, torch_model)

        for batch_size in BATCH_SIZES:
            x_np = np.random.randn(batch_size, input_dim).astype(np.float32)

            # Forward passes
            with torch.no_grad():
                torch_out = torch_model(torch.from_numpy(x_np)).numpy()
            tf_out = tf_model(x_np, training=False).numpy()

            # Basic shape & numerical check
            assert torch_out.shape == tf_out.shape
            assert np.allclose(torch_out, tf_out, atol=1e-5)
