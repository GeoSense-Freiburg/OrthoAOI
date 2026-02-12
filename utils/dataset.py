"""Datasets for AOI segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class AOICOCOSegmentationDataset(Dataset):
    """COCO-style dataset returning orthomosaic image tiles and binary AOI masks."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        tile_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.tile_size = tile_size

        with self.annotations_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images: List[Dict[str, Any]] = coco.get("images", [])
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            image_id = ann["image_id"]
            self.annotations_by_image.setdefault(image_id, []).append(ann)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.tile_size, self.tile_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.tile_size, self.tile_size), interpolation=InterpolationMode.NEAREST
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def _polygons_to_mask(
        self, segmentation: Sequence[Sequence[float]], height: int, width: int
    ) -> np.ndarray:
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(points, outline=1, fill=1)
        return np.array(mask_img, dtype=np.float32)

    def _rle_to_mask(self, rle: Dict[str, Any], height: int, width: int) -> np.ndarray:
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError(
                "pycocotools is required for RLE-encoded segmentations. "
                "Install it with `pip install pycocotools`."
            ) from exc

        if isinstance(rle.get("counts"), list):
            encoded = mask_utils.frPyObjects(rle, height, width)
        else:
            encoded = rle
        decoded = mask_utils.decode(encoded)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_info = self.images[idx]
        image_path = self.images_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        mask = np.zeros((height, width), dtype=np.float32)
        for ann in self.annotations_by_image.get(image_info["id"], []):
            segmentation = ann.get("segmentation")
            if isinstance(segmentation, list):
                ann_mask = self._polygons_to_mask(segmentation, height, width)
            elif isinstance(segmentation, dict):
                ann_mask = self._rle_to_mask(segmentation, height, width)
            else:
                continue
            mask = np.maximum(mask, ann_mask)

        image_tensor = self.image_transform(image)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_tensor = self.mask_transform(mask_pil)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor


class InferenceImageDataset(Dataset):
    """Image-only dataset for inference; returns (image, path)."""

    def __init__(
        self,
        images_dir: str | Path,
        tile_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        self.images_dir = Path(images_dir)
        self.tile_size = tile_size
        self.image_paths = sorted(
            [
                p
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
                for p in self.images_dir.glob(ext)
            ]
        )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.tile_size, self.tile_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        return image_tensor, str(image_path)
