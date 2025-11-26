from typing import Dict

import numpy as np


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean(np.abs(a - b)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a).ravel()
    b_flat = np.asarray(b).ravel()
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def summarize_output_diff(torch_out: np.ndarray, tf_out: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(torch_out, tf_out)
    cos_sim = cosine_similarity(torch_out, tf_out)
    return {
        "mae": mae,
        "cosine_similarity": cos_sim,
        "cosine_distance": 1.0 - cos_sim,
    }
