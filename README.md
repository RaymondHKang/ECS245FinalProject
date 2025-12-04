# Differential Testing Between PyTorch and TensorFlow

This project provides a **deterministic, extensible, and architecture-agnostic testbed** for differential testing between **PyTorch** and **TensorFlow** implementations of equivalent neural networks.  
It focuses on verifying numerical consistency across frameworks under a wide variety of controlled conditions.

The system currently supports:

- Flexible MLP architectures (arbitrary depth and width)
- Multiple activation functions
- Deterministic reproducibility
- TF → PyTorch weight synchronization
- Robust numerical comparison metrics
- Sweeps across seeds, batch sizes, activations, architectures, and dtypes
- Forward-pass equivalence testing, performance testing, and edge-case evaluation

This repository is under active development.

---

# 1. Project Overview

Different ML frameworks implement operations with subtly different numerical kernels, optimization paths, or approximations.  
This project constructs a **rigorous differential testing harness** to detect such discrepancies.

So far, the system supports **forward-pass differential testing** across a wide variety of conditions.

---

# 2. Implemented Features

This section documents all functionality currently available.

---

## 2.1 Flexible MLP Architecture (TF & PyTorch)

Both frameworks now support:

- Arbitrary hidden layer lists:
  - `[64]`, `[64, 64]`, `[128, 64, 32]`, etc.
- Per-layer activation functions:
  - `relu`
  - `tanh`
  - `sigmoid`
  - `gelu`
  - `elu`
- Identical architecture specification on both frameworks
- Full backward compatibility with the original single-layer MLP

This enables systematic testing of a wide variety of architectures.

---

## 2.2 Deterministic Reproducibility

The utility `set_global_seed(seed)` synchronizes:

- Python `random`
- NumPy
- PyTorch (CPU deterministic mode)
- TensorFlow
- TF/Keras floatx configuration

All experiments enforce:`CUDA_VISIBLE_DEVICES = "-1"`
ensuring **CPU-only deterministic execution**.

Synthetic inputs are saved and reused for cross-experiment consistency.

---

## 2.3 Centralized TF → PyTorch Weight Synchronization

The new `utils/sync_weights.py`:

- Locates all Keras `Dense` layers
- Locates all PyTorch `Linear` layers
- Matches them layer-by-layer
- Transposes weight matrices (TF: `[in, out]`, Torch: `[out, in]`)
- Copies biases
- Ensures dtype alignment (`float32` or `float64`)

All experiments use this centralized logic (no duplication).

---

## 2.4 Robust Output Comparison Metrics

`utils/compare_outputs.py` implements:

- **Mean Absolute Error (MAE)**
- **Cosine similarity** (clamped to `[−1,1]`)
- **Cosine distance**

Numerical issues such as:

- near-zero norms
- NaNs
- minor rounding overflow beyond 1.0

are handled safely.

This ensures stable, framework-agnostic comparison.

---

# 3. Experimental Capabilities

These scripts allow systematic evaluation of PyTorch vs TensorFlow differences.

---

## 3.1 Forward Pass Equivalence (`run_forward_test.py`)

- Builds equivalent TF and Torch models
- Synchronizes weights
- Runs a forward pass on shared synthetic input
- Computes MAE + cosine similarity
- Confirms architectural equivalence

---

## 3.2 Performance Comparison (`run_performance_test.py`)

- Measures average runtime across multiple runs
- Measures memory delta (RSS)
- Evaluates CPU-side efficiency of both frameworks

---

## 3.3 Edge-Case Behavior (`run_edge_case_test.py`)

Tests extreme values:

- `0`
- `1e-12`
- `1e12`
- `nan`
- `inf`
- `-inf`

Logs:

- model success/failure
- any numerical instability
- output summary statistics

---

## 3.4 Forward Sweep (Seeds × Batch Sizes)

`run_forward_sweep.py` verifies stability across:

- multiple random seeds
- multiple batch sizes
- consistent TF/PyTorch agreement

---

## 3.5 Dtype Sweep (`float32` vs `float64`)

`run_dtype_sweep.py` verifies consistency with:

- high precision
- mixed precision patterns
- seed/batch stability across dtypes

---

## 3.6 Architecture × Activation × Dtype Sweep

`run_arch_activation_sweep.py` performs broad testing across:

- arbitrary multi-layer architectures
- per-layer activation patterns
- float32 & float64
- multiple seeds
- multiple batch sizes

This provides a comprehensive picture of framework agreement.

---

## 3.7 Gradient Comparison (Single Architecture)

`run_gradient_test.py`  
- Builds equivalent TF and Torch models for a simple, well-behaved architecture  
- Synchronizes weights  
- Computes gradients of MSE loss with respect to all parameters  
- Aligns weight gradient layouts (Dense vs Linear)  
- Reports global and layer-wise gradient MAE and cosine similarity

---

## 3.8 Gradient Sweep (arch × activation × dtype × seed × batch)

`run_gradient_sweep.py`  
- Sweeps over multiple architectures, activations, dtypes, seeds, and batch sizes  
- For each configuration:
  - Computes TF and Torch gradients
  - Aligns and compares them (MAE, cosine similarity)
  - Records gradient norms and NaN/Inf flags  
- Writes results to `results/gradient_sweep.csv` for later analysis

---

# 4. Running the Experiments
## 4.1 Run Everything (Full Experiment Suite)
To execute the entire differential testing pipeline — including forward tests, performance profiling, edge-case analysis, architecture sweeps, dtype sweeps, and gradient sweeps:
```bash
python -m main
```
---

## 4.2 Individual Experiments
### Forward Pass Comparison
Basic numerical equivalence test on a single batch:
```bash
python -m experiments.run_forward_test
```
### Performance test
Measures CPU runtime and memory usage for PyTorch vs TensorFlow:
```bash
python -m experiments.run_performance_test
```
### Edge-case behavior
Evaluates stability under extreme values (0, tiny, huge, NaN, ±Inf):
```bash
python -m experiments.run_edge_case_test
```
### Forward sweep (Seeds × Batch Sizes)
Systematic forward-pass checks across multiple seeds and batch sizes:
```bash
python -m experiments.run_forward_sweep
```
### Dtype Sweep (float32 vs float64)
Verifies numerical consistency across different data types:
```bash
python -m experiments.run_dtype_sweep
```
### Architecture × Activation × Dtype Sweep
Tests deeper networks and activation functions for forward-pass consistency:
```bash
python -m experiments.run_arch_activation_sweep
```
Gradient Sweep (Backward-Pass Differential Test)
Compares TF vs PyTorch gradients across architectures, activations, dtypes, seeds, and batch sizes:
```bash
python -m experiments.run_gradient_sweep
```

---

# 5. Configuration

`config/settings.yaml` controls:

- model dimensions
- batch sizes
- seeds
- paths for storing synthetic inputs
- dtype configuration
- reproducibility settings

Experiments dynamically load and apply these settings.

---
