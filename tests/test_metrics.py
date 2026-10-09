import numpy as np
import pytest

from automotive_int8.runtime_utils import time_inference


class FakeInput:
    name = "images"


class FakeSession:
    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, feed):
        assert "images" in feed
        return [np.zeros((1, 1), dtype=np.float32)]


def test_time_inference_returns_expected_metrics():
    metrics = time_inference(FakeSession(), np.zeros((1, 3, 8, 8), dtype=np.float32),
                             warmup=1, runs=5)
    assert metrics["runs"] == 5
    assert metrics["latency_median_ms"] >= 0
    assert metrics["latency_p95_ms"] >= 0
    assert metrics["throughput_images_per_sec"] > 0


def test_invalid_run_count_raises():
    with pytest.raises(ValueError):
        time_inference(FakeSession(), np.zeros((1, 3, 8, 8), dtype=np.float32),
                       warmup=0, runs=0)
