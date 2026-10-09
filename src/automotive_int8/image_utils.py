from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: str | Path) -> list[Path]:
    """Return supported image files in a directory, sorted deterministically."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def preprocess_image(path: str | Path, image_size: int = 640) -> np.ndarray:
    """
    Simple square resize and RGB normalization for YOLO-style ONNX inputs.

    Returns a float32 tensor shaped [1, 3, image_size, image_size] in [0, 1].
    For report-grade accuracy evaluation, use the exact preprocessing expected
    by the model/export pipeline (including letterboxing if applicable).
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    with Image.open(path) as im:
        im = im.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def preprocess_many(paths: Iterable[str | Path], image_size: int = 640) -> list[np.ndarray]:
    return [preprocess_image(path, image_size) for path in paths]
