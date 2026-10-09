from pathlib import Path
import time
import numpy as np
import onnxruntime as ort


def create_session(model_path: str | Path, intra_op_threads: int = 0) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    # CPU provider keeps comparisons portable. Pass a different provider explicitly
    # only after confirming it is installed and supported by your ONNX Runtime build.
    return ort.InferenceSession(str(model_path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def run_model(session: ort.InferenceSession, input_tensor: np.ndarray):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor})


def time_inference(session: ort.InferenceSession, input_tensor: np.ndarray,
                   warmup: int = 10, runs: int = 100) -> dict:
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")
    for _ in range(warmup):
        run_model(session, input_tensor)
    samples_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        run_model(session, input_tensor)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "runs": runs,
        "latency_mean_ms": float(np.mean(arr)),
        "latency_median_ms": float(np.median(arr)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_min_ms": float(np.min(arr)),
        "throughput_images_per_sec": float(1000.0 / np.mean(arr)),
    }
