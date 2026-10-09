#!/usr/bin/env python3
"""Export an Ultralytics YOLO checkpoint to a floating-point ONNX model."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n.pt", help="Ultralytics model/checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    exported = model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=False)
    exported_path = Path(exported)
    target = output_dir / exported_path.name
    if exported_path.resolve() != target.resolve():
        target.write_bytes(exported_path.read_bytes())
    print(f"Exported ONNX model: {target}")
    print("Record the exact weights, Ultralytics version, image size, and export settings.")


if __name__ == "__main__":
    main()
