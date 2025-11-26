"""Orchestrator to run all experiments from one entry point."""

from experiments.run_forward_test import main as forward_main
from experiments.run_performance_test import main as perf_main
from experiments.run_edge_case_test import main as edge_main


def main():
    print("Running forward comparison...")
    forward_main()
    print("\nRunning performance benchmark...")
    perf_main()
    print("\nRunning edge-case stability tests...")
    edge_main()


if __name__ == "__main__":
    main()
