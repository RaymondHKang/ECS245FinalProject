import os
import time
from typing import Callable, Tuple

import psutil
import gc


def get_memory_mb() -> float:
    """Return Resident Set Size (RSS) of current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def time_forward_pass(
    func: Callable[[], None],
    num_runs: int = 100,
) -> Tuple[float, float]:
    """
    Time a forward pass function multiple times.

    Args:
        func: callable that executes a full forward pass (no args).
        num_runs: number of repetitions.

    Returns:
        (avg_time_sec, std_time_sec)
    """
    times = []
    # Warmup
    func()
    for _ in range(num_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    avg = sum(times) / len(times)
    var = sum((t - avg) ** 2 for t in times) / max(len(times) - 1, 1)
    std = var ** 0.5
    return avg, std


def measure_memory_for_call(func: Callable[[], None]) -> float:
    """Approximate memory delta (MB) for a single call to func, using RSS diff."""
    gc.collect()
    before = get_memory_mb()
    func()
    gc.collect()
    after = get_memory_mb()
    return after - before
