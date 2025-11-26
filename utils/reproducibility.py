import os
import random

import numpy as np
import torch
import tensorflow as tf


def set_global_seed(seed: int = 0) -> None:
    """
    Set random seeds and deterministic flags for numpy, Python, PyTorch, and TensorFlow.
    Note: full determinism is not guaranteed across hardware/backends.
    """
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # TensorFlow
    tf.random.set_seed(seed)
    # For TF deterministic ops (esp. on GPU) if installed:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


def load_yaml_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
