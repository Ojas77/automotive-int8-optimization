#!/usr/bin/env python3
"""Apply ONNX Runtime static QDQ INT8 quantization to an ONNX model."""
import argparse
from pathlib import Path
import sys

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from automotive_int8.image_utils import list_images, preprocess_image  # noqa: E402


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed images to ONNX Runtime's static quantizer."""

    def __init__(self, image_paths, input_name, image_size):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.image_size = image_size
        self._index = 0

    def get_next(self):
        if self._index >= len(self.image_paths):
            return None
        path = self.image_paths[self._index]
        self._index += 1
        return {self.input_name: preprocess_image(path, self.image_size)}

    def rewind(self):
        self._index = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="FP32 ONNX model path")
    parser.add_argument("--calibration-dir", required=True, help="Directory of calibration images")
    parser.add_argument("--output", default="artifacts/model_int8.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    image_paths = list_images(args.calibration_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in {args.calibration_dir}")
    if args.max_images < 1:
        raise ValueError("--max-images must be >= 1")
    image_paths = image_paths[:args.max_images]

    import onnxruntime as ort
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    shape = model_input.shape
    # This starter expects a static NCHW float input matching --imgsz.
    if len(shape) != 4 or shape[1] not in (3, "3", None):
        raise ValueError(f"Expected a 4D NCHW image input, got {shape}")
    if isinstance(shape[2], int) and shape[2] != args.imgsz:
        raise ValueError(f"Model height is {shape[2]}, but --imgsz={args.imgsz}")
    if isinstance(shape[3], int) and shape[3] != args.imgsz:
        raise ValueError(f"Model width is {shape[3]}, but --imgsz={args.imgsz}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageCalibrationReader(image_paths, input_name, args.imgsz)
    print(f"Calibrating with {len(image_paths)} images; input={input_name}, shape={shape}")
    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=__import__(
            "onnxruntime.quantization", fromlist=["CalibrationMethod"]
        ).CalibrationMethod.MinMax,
        per_channel=True,
        reduce_range=False,
    )
    print(f"Quantized model saved: {output_path}")
    print("Note: operator support and speedups depend on your ONNX Runtime build and hardware.")


if __name__ == "__main__":
    main()
