# Differential Testing of PyTorch and TensorFlow

This project performs differential testing between equivalent MLP models
implemented in PyTorch and TensorFlow. It compares:

- **Correctness**: numerical agreement of forward outputs.
- **Performance**: average forward time and rough memory usage.
- **Numerical stability**: behavior on edge-case inputs (0, tiny, huge, NaN, inf).

## Structure

- `config/` – global settings (model dims, seeds, paths)
- `models/` – PyTorch and TensorFlow MLP definitions
- `utils/` – reproducibility, output comparison, performance, edge-case generators
- `experiments/` – scripts to run different experiments
- `tests/` – quick sanity checks for alignment/correctness

## Quickstart
Python Version=3.11
```bash
pip install -r requirements.txt

# Run forward correctness test
python -m experiments.run_forward_test

# Run performance benchmark
python -m experiments.run_performance_test

# Run edge-case stability test
python -m experiments.run_edge_case_test

# (Optional) run tests
pytest
```
# ECS245FinalProject
