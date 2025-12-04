"""Orchestrator to run the full differential testing experiment suite."""

from experiments.run_forward_test import main as forward_main
from experiments.run_performance_test import main as perf_main
from experiments.run_edge_case_test import main as edge_main
from experiments.run_forward_sweep import main as forward_sweep_main
from experiments.run_dtype_sweep import main as dtype_sweep_main
from experiments.run_arch_activation_sweep import main as arch_act_sweep_main
from experiments.run_gradient_test import main as grad_basic_main
from experiments.run_gradient_sweep import main as grad_sweep_main


def main():
    print("=== 1) Forward comparison (single run) ===")
    forward_main()

    print("\n=== 2) Performance benchmark ===")
    perf_main()

    print("\n=== 3) Edge-case stability tests ===")
    edge_main()

    print("\n=== 4) Forward sweep (seeds × batch sizes) ===")
    forward_sweep_main()

    print("\n=== 5) Dtype sweep (float32 vs float64) ===")
    dtype_sweep_main()

    print("\n=== 6) Architecture × activation × dtype sweep (forward) ===")
    arch_act_sweep_main()

    print("\n=== 7) Gradient comparison (single architecture sanity check) ===")
    grad_basic_main()

    print("\n=== 8) Gradient sweep (arch × activation × dtype × seed × batch) ===")
    grad_sweep_main()

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
