from typing import List

import numpy as np


def generate_edge_case_values() -> List[float]:
    return [0.0, 1e-12, 1e12, np.nan, np.inf, -np.inf]


def generate_edge_case_batch(
    value: float,
    batch_size: int,
    input_dim: int,
    dtype=np.float32,
) -> np.ndarray:
    """Create a batch where every entry is `value`."""
    return np.full((batch_size, input_dim), value, dtype=dtype)
