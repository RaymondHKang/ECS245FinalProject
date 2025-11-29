from typing import Dict

import numpy as np

def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        # If both are (near) zero, treat them as identical; else, define sim = 0
        if np.allclose(a, b, atol=1e-12):
            return 1.0
        else:
            return 0.0

    sim = float(np.dot(a, b) / denom)

    # Clamp to valid range [-1, 1] to avoid 1.00000005, etc.
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0

    return sim

def summarize_output_diff(torch_output: np.ndarray, tf_output: np.ndarray) -> dict:
    a = np.asarray(torch_output, dtype=np.float64)
    b = np.asarray(tf_output, dtype=np.float64)

    mae = mean_absolute_error(a, b)
    cos_sim = cosine_similarity(a, b)
    cos_dist = 1.0 - cos_sim

    return {
        "mae": mae,
        "cosine_similarity": cos_sim,
        "cosine_distance": cos_dist,
    }