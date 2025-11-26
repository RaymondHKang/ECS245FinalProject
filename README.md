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

```bash
pip install -r requirements.txt

# Run forward correctness test
python experiments/run_forward_test.py

# Run performance benchmark
python experiments/run_performance_test.py

# Run edge-case stability test
python experiments/run_edge_case_test.py

# (Optional) run tests
pytest
```
# ECS245FinalProject
