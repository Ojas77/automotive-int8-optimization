#!/usr/bin/env python3
"""Plot benchmark CSV results."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_benchmarks(frame: pd.DataFrame, output_path: str | Path):
    if frame.empty or "model" not in frame or "latency_median_ms" not in frame:
        raise ValueError("Benchmark data must include model and latency_median_ms columns")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(frame["model"], frame["latency_median_ms"])
    ax.set_ylabel("Median inference latency (ms)")
    ax.set_title("FP32 vs INT8 ONNX inference latency")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/benchmark.csv")
    parser.add_argument("--output", default="results/latency_comparison.png")
    args = parser.parse_args()
    plot_benchmarks(pd.read_csv(args.csv), args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
