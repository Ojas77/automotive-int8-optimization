from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from automotive_int8.image_utils import list_images, preprocess_image


def test_preprocess_image_shape_and_range(tmp_path: Path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), color=(255, 128, 0)).save(path)
    tensor = preprocess_image(path, image_size=32)
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_list_images_sorted_and_filters(tmp_path: Path):
    Image.new("RGB", (8, 8)).save(tmp_path / "b.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    (tmp_path / "ignore.txt").write_text("x")
    assert [p.name for p in list_images(tmp_path)] == ["a.png", "b.jpg"]


def test_missing_directory_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "missing")
