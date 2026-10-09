#!/usr/bin/env python3
"""Benchmark ONNX models with identical preprocessed inputs."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from automotive_int8.image_utils import list_images, preprocess_image  # noqa: E402
from automotive_int8.runtime_utils import create_session, time_inference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=20,
                        help="Number of images to sample for benchmarking")
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    images = list_images(args.images_dir)
    if not images:
        raise ValueError(f"No supported images found in {args.images_dir}")
    images = images[:max(1, args.max_images)]
    tensors = [preprocess_image(p, args.imgsz) for p in images]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    process = psutil.Process()

    for model_path in args.models:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        session = create_session(model_path, args.intra_op_threads)
        # Use first image for repeated timing, and rotate through the selected
        # images once for an additional representative pass.
        timings = time_inference(session, tensors[0], args.warmup, args.runs)
        for tensor in tensors:
            session.run(None, {session.get_inputs()[0].name: tensor})
        rows.append({
            "model": model_path.name,
            "model_path": str(model_path),
            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            "execution_provider": session.get_providers()[0],
            "image_size": args.imgsz,
            "validation_images_used": len(tensors),
            "process_rss_mb_after_run": process.memory_info().rss / (1024 * 1024),
            **timings,
        })
        print(f"{model_path.name}: median={timings['latency_median_ms']:.3f} ms, "
              f"p95={timings['latency_p95_ms']:.3f} ms")

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark.csv"
    json_path = output_dir / "benchmark.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path} and {json_path}")
    try:
        from scripts.plot_results import plot_benchmarks
        plot_benchmarks(frame, output_dir / "latency_comparison.png")
        print(f"Wrote {output_dir / 'latency_comparison.png'}")
    except Exception as exc:
        print(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()
