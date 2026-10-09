# Automotive Computer Vision: FP32 vs INT8 Quantization

A reproducible, Git-ready starter project for evaluating post-training INT8 quantization of a YOLO object detector exported to ONNX.

The project measures the accuracy/performance trade-off between:
- **FP32 ONNX**: the exported floating-point baseline.
- **INT8 ONNX**: post-training static quantization using ONNX Runtime.

It is designed as an educational portfolio project for edge AI / automotive computer vision. It does **not** claim to be safety-certified or suitable for use in a real vehicle.

## What you get

- Export a pretrained YOLO model to ONNX.
- Prepare a small calibration/validation image set.
- Quantize the ONNX model to INT8 using ONNX Runtime.
- Benchmark latency, throughput, model size, and peak process memory (where supported).
- Optional mAP evaluation using Ultralytics validation.
- CSV and JSON result outputs plus a plot.
- Reproducible CLI commands and configuration.

## Quick start

### 1. Requirements

- Python 3.10 or 3.11 recommended.
- CPU works for export/quantization/inference; a GPU is optional.
- Internet access is needed the first time to download the pretrained weights and dataset.
- Quantization speedups depend on CPU/runtime support. INT8 may be slower on hardware without efficient INT8 kernels.

### 2. Create environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Export FP32 ONNX model

```bash
python scripts/export_model.py --weights yolo26n.pt --imgsz 640
```

The script uses Ultralytics to download/load the weights and exports `artifacts/yolo26n.onnx`. You can change `--weights` to another compatible Ultralytics detection checkpoint.

### 4. Get calibration/validation images

For a reproducible small smoke test, download COCO8 via Ultralytics:

```bash
yolo detect val model=yolo26n.pt data=coco8.yaml imgsz=640
```

The export/quantization scripts accept a directory of images. Point `--calibration-dir` to a folder containing `.jpg`, `.jpeg`, `.png`, or `.bmp` files. Use images representative of the environment you care about. For a meaningful study, use a held-out validation set rather than using the same images for calibration and final evaluation.

You may use a public road-scene dataset such as BDD100K or COCO images, subject to its terms. This repository does not redistribute datasets.

### 5. Quantize FP32 ONNX to INT8

```bash
python scripts/quantize_model.py \
  --model artifacts/yolo26n.onnx \
  --calibration-dir data/calibration \
  --output artifacts/yolo26n_int8.onnx \
  --imgsz 640 \
  --max-images 100
```

The static quantization pipeline uses representative images to calibrate activation ranges. The default is QDQ quantization. If the model/runtime has unsupported operators, quantization can fail or leave some operators in floating point; inspect the logs and resulting model.

**Important:** `data/calibration/` must contain images before running this command. Do not use your final test set as calibration data.

### 6. Benchmark both models

```bash
python scripts/benchmark.py \
  --models artifacts/yolo26n.onnx artifacts/yolo26n_int8.onnx \
  --images-dir data/validation \
  --imgsz 640 \
  --warmup 10 \
  --runs 100 \
  --output-dir results
```

Outputs:
- `results/benchmark.csv`
- `results/benchmark.json`
- `results/latency_comparison.png`

Benchmarking uses ONNX Runtime and reports raw model inference latency (preprocessed input tensor to model output), not end-to-end camera latency. Run on the same machine, provider, image size, and thread settings for a fair comparison.

### 7. Optional accuracy evaluation

For a proper object-detection accuracy comparison, evaluate each ONNX model on the same labeled dataset using Ultralytics:

```bash
yolo detect val model=artifacts/yolo26n.onnx data=coco8.yaml imgsz=640
yolo detect val model=artifacts/yolo26n_int8.onnx data=coco8.yaml imgsz=640
```

Use a larger, held-out dataset for report-quality results. Record the exact dataset, version, model, runtime, and hardware. Do not report accuracy numbers unless you actually measured them.

## Suggested research questions

1. How much does INT8 quantization reduce file size?
2. Does latency improve on your CPU? Does it change with `CPUExecutionProvider` versus another supported provider?
3. What accuracy degradation occurs on a held-out road-scene validation set?
4. How does calibration set size (e.g. 25, 50, 100, 250 images) affect accuracy?
5. Which classes or conditions (night, rain, small objects, occlusion) are most affected?

## Repository layout

```text
automotive-int8-optimization/
├── configs/default.yaml
├── data/
│   ├── calibration/.gitkeep
│   └── validation/.gitkeep
├── artifacts/.gitkeep
├── results/.gitkeep
├── scripts/
│   ├── export_model.py
│   ├── quantize_model.py
│   ├── benchmark.py
│   └── plot_results.py
├── src/automotive_int8/
│   ├── __init__.py
│   ├── image_utils.py
│   └── runtime_utils.py
├── tests/
│   ├── test_image_utils.py
│   └── test_metrics.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

## Reproducibility checklist

- [ ] Record OS, CPU/GPU, Python version, and ONNX Runtime version.
- [ ] Use the exact same validation images for FP32 and INT8.
- [ ] Separate calibration images from validation images.
- [ ] Include warm-up runs and enough repeated measurements.
- [ ] Report median and p95 latency, not only a single run.
- [ ] Measure accuracy and latency independently.
- [ ] State whether measurements use CPU, CUDA, or another execution provider.
- [ ] Document failures, unsupported operators, and limitations.

## Example resume bullets (fill in real measurements only)

- Exported a YOLO object detector to ONNX and implemented post-training static INT8 quantization using ONNX Runtime.
- Compared FP32 and INT8 inference using [dataset] on [hardware], measuring [actual size reduction], [actual latency change], and [actual mAP change].
- Built a reproducible benchmarking pipeline that exports results to CSV/JSON and plots the accuracy/performance trade-off.

## Safety and limitations

This is a research/portfolio prototype. It is not an ADAS component and must not be connected to vehicle control. Benchmark results are hardware- and dataset-dependent. Check licenses for model weights and datasets before redistribution or commercial use.
