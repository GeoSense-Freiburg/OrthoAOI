"""Shared utilities for CLI workflows."""

from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
from PIL import Image


def resolve_weights(weights_value: str | None, models_dir: str) -> str | None:
    if not weights_value:
        return None
    parsed = urllib.parse.urlparse(weights_value)
    if parsed.scheme in {"http", "https"}:
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        filename = Path(parsed.path).name or "dinov3_weights.pth"
        destination = models_path / filename
        if not destination.exists():
            urllib.request.urlretrieve(weights_value, destination)  # noqa: S310
        weights_path = destination
    else:
        weights_path = Path(weights_value)

    if not weights_path.exists():
        raise FileNotFoundError(f"Backbone weights not found: {weights_path}")
    if weights_path.stat().st_size == 0:
        raise ValueError(f"Backbone weights file is empty: {weights_path}")

    return str(weights_path)


def flatten_predictions(predictions: Iterable[Any]) -> List[dict]:
    flattened: List[dict] = []
    for batch in predictions:
        if isinstance(batch, dict):
            flattened.append(batch)
        else:
            flattened.extend(batch)
    return flattened


def save_prediction_masks(
    predictions: Iterable[Any], output_dir: str | Path, threshold: float = 0.5
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for item in flatten_predictions(predictions):
        paths = item["paths"]
        masks = item["masks"]
        for path, mask in zip(paths, masks):
            mask_np = mask.squeeze().cpu().numpy()
            mask_img = Image.fromarray(
                (mask_np > threshold).astype(np.uint8) * 255, mode="L"
            )
            output_file = output_path / f"{Path(path).stem}_mask.png"
            mask_img.save(output_file)
