"""Datasets and LightningDataModule for AOI segmentation."""

from __future__ import annotations

from pathlib import Path
import csv
import logging
import random
import time
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode

logger = logging.getLogger(__name__)

_IMAGE_EXTS = IMG_EXTENSIONS


def _list_image_paths(images_dir: Path) -> List[Path]:
    return sorted([p for ext in _IMAGE_EXTS for p in images_dir.rglob(f"*{ext}")])


def _open_with_retries(
    path: Path, mode: str, io_retries: int, io_retry_delay: float
) -> Image.Image:
    last_err: Exception | None = None
    for attempt in range(io_retries + 1):
        try:
            with Image.open(path) as img:
                return img.convert(mode)
        except (BlockingIOError, OSError, UnidentifiedImageError) as exc:
            last_err = exc
            if attempt < io_retries:
                time.sleep(io_retry_delay * (2 ** attempt))
    assert last_err is not None
    raise last_err


def _resample_index(idx: int, limit: int, length: int) -> int:
    return (idx + 1 + random.randrange(0, limit + 1)) % length


def _build_image_transform(
    tile_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((tile_size, tile_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _build_mask_transform(tile_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (tile_size, tile_size), interpolation=InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ]
    )


class AOIMaskSegmentationDataset(Dataset):
    """Paired image + mask dataset for AOI segmentation."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path | None,
        tile_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        io_retries: int = 3,
        io_retry_delay: float = 0.1,
        io_resample_on_error: bool = True,
        io_resample_limit: int = 10,
        skip_missing_pairs: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir is not None else None
        self.tile_size = tile_size
        self.samples: List[Tuple[Path, Path]] = []
        self.image_paths: List[Path] = []
        self.io_retries = io_retries
        self.io_retry_delay = io_retry_delay
        self.io_resample_on_error = io_resample_on_error
        self.io_resample_limit = io_resample_limit
        self.skip_missing_pairs = skip_missing_pairs

        if self.images_dir.is_file():
            self.samples = self._load_pairs_from_csv(self.images_dir)
            self.image_paths = [pair[0] for pair in self.samples]
        else:
            if masks_dir is None:
                raise ValueError("masks_dir is required when images_dir is a directory.")
            self.image_paths = _list_image_paths(self.images_dir)
            if not self.image_paths:
                raise ValueError(f"No images found in {self.images_dir}.")

            skipped_missing = 0
            for image_path in self.image_paths:
                try:
                    mask_path = self._match_mask(image_path)
                except FileNotFoundError:
                    if not self.skip_missing_pairs:
                        raise
                    skipped_missing += 1
                    continue
                self.samples.append((image_path, mask_path))

            if skipped_missing:
                logger.warning(
                    "Skipped %d image(s) without matching masks in %s.",
                    skipped_missing,
                    self.masks_dir,
                )

            if not self.samples:
                raise ValueError(
                    "No valid image/mask pairs found after filtering missing masks in "
                    f"{self.images_dir} and {self.masks_dir}."
                )

        self.image_transform = _build_image_transform(self.tile_size, mean, std)
        self.mask_transform = _build_mask_transform(self.tile_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _match_mask(self, image_path: Path) -> Path:
        rel_parent = Path(".")
        try:
            rel_parent = image_path.relative_to(self.images_dir).parent
        except ValueError:
            pass
        mask_parent = self.masks_dir / rel_parent

        candidate = mask_parent / image_path.name
        if candidate.exists():
            return candidate
        for ext in _IMAGE_EXTS:
            candidate = mask_parent / f"{image_path.stem}{ext}"
            if candidate.exists():
                return candidate
        for suffix in ("_mask", "_masks"):
            for ext in _IMAGE_EXTS:
                candidate = mask_parent / f"{image_path.stem}{suffix}{ext}"
                if candidate.exists():
                    return candidate
        for prefix in ("mask_", "masks_"):
            for ext in _IMAGE_EXTS:
                candidate = mask_parent / f"{prefix}{image_path.stem}{ext}"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"No mask found for image {image_path.name} in {mask_parent}."
        )

    def _resolve_csv_path(self, raw_path: str, csv_path: Path) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        csv_relative = csv_path.parent / candidate
        if csv_relative.exists():
            return csv_relative
        return candidate

    def _load_pairs_from_csv(self, csv_path: Path) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        skipped_missing = 0
        with csv_path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and "image" in header and "mask" in header:
                img_idx = header.index("image")
                mask_idx = header.index("mask")
                for row in reader:
                    if not row:
                        continue
                    image_path = self._resolve_csv_path(row[img_idx], csv_path)
                    mask_path = self._resolve_csv_path(row[mask_idx], csv_path)
                    if self.skip_missing_pairs and (
                        not image_path.exists() or not mask_path.exists()
                    ):
                        skipped_missing += 1
                        continue
                    pairs.append((image_path, mask_path))
            else:
                if header:
                    # Treat header as first row if it doesn't match expected columns.
                    if len(header) >= 2:
                        image_path = self._resolve_csv_path(header[0], csv_path)
                        mask_path = self._resolve_csv_path(header[1], csv_path)
                        if self.skip_missing_pairs and (
                            not image_path.exists() or not mask_path.exists()
                        ):
                            skipped_missing += 1
                        else:
                            pairs.append((image_path, mask_path))
                for row in reader:
                    if not row:
                        continue
                    if len(row) < 2:
                        continue
                    image_path = self._resolve_csv_path(row[0], csv_path)
                    mask_path = self._resolve_csv_path(row[1], csv_path)
                    if self.skip_missing_pairs and (
                        not image_path.exists() or not mask_path.exists()
                    ):
                        skipped_missing += 1
                        continue
                    pairs.append((image_path, mask_path))
        if not pairs:
            raise ValueError(f"No image/mask pairs found in {csv_path}.")
        if skipped_missing:
            logger.warning(
                "Skipped %d CSV row(s) with missing files in %s.",
                skipped_missing,
                csv_path,
            )
        return pairs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        attempts = 0
        while True:
            image_path, mask_path = self.samples[idx]
            try:
                image = _open_with_retries(
                    image_path, "RGB", self.io_retries, self.io_retry_delay
                )
                mask = _open_with_retries(
                    mask_path, "L", self.io_retries, self.io_retry_delay
                )
                image_tensor = self.image_transform(image)
                mask_tensor = self.mask_transform(mask)
                mask_tensor = (mask_tensor > 0.5).float()
                return image_tensor, mask_tensor
            except (BlockingIOError, OSError, UnidentifiedImageError):
                if not self.io_resample_on_error or attempts >= self.io_resample_limit:
                    raise
                attempts += 1
                idx = _resample_index(idx, self.io_resample_limit, len(self.samples))


class InferenceImageDataset(Dataset):
    """Image-only dataset for inference; returns (image, path)."""

    def __init__(
        self,
        images_dir: str | Path,
        tile_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        io_retries: int = 3,
        io_retry_delay: float = 0.1,
        io_resample_on_error: bool = True,
        io_resample_limit: int = 10,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.tile_size = tile_size
        self.io_retries = io_retries
        self.io_retry_delay = io_retry_delay
        self.io_resample_on_error = io_resample_on_error
        self.io_resample_limit = io_resample_limit
        self.image_paths = _list_image_paths(self.images_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}.")
        self.image_transform = _build_image_transform(self.tile_size, mean, std)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        attempts = 0
        while True:
            image_path = self.image_paths[idx]
            try:
                image = _open_with_retries(
                    image_path, "RGB", self.io_retries, self.io_retry_delay
                )
                image_tensor = self.image_transform(image)
                return image_tensor, str(image_path)
            except (BlockingIOError, OSError, UnidentifiedImageError):
                if not self.io_resample_on_error or attempts >= self.io_resample_limit:
                    raise
                attempts += 1
                idx = _resample_index(idx, self.io_resample_limit, len(self.image_paths))


class AOIDataModule(pl.LightningDataModule):
    """DataModule providing train/val/predict loaders."""

    def __init__(
        self,
        train_images: str | None,
        train_masks: str | None,
        val_images: str | None,
        val_masks: str | None,
        predict_images: str | None,
        tile_size: int,
        batch_size: int,
        num_workers: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        io_retries: int = 3,
        io_retry_delay: float = 0.1,
        io_resample_on_error: bool = True,
        io_resample_limit: int = 10,
        skip_missing_pairs: bool = True,
    ) -> None:
        super().__init__()
        self.train_images = train_images
        self.train_masks = train_masks
        self.val_images = val_images
        self.val_masks = val_masks
        self.predict_images = predict_images
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        self.io_retries = io_retries
        self.io_retry_delay = io_retry_delay
        self.io_resample_on_error = io_resample_on_error
        self.io_resample_limit = io_resample_limit
        self.skip_missing_pairs = skip_missing_pairs

        self.train_dataset: AOIMaskSegmentationDataset | None = None
        self.val_dataset: AOIMaskSegmentationDataset | None = None
        self.predict_dataset: InferenceImageDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if not self.train_images or not self.val_images:
                raise ValueError("Training requires train_images and val_images.")
            self.train_dataset = AOIMaskSegmentationDataset(
                images_dir=self.train_images,
                masks_dir=self.train_masks,
                tile_size=self.tile_size,
                mean=self.mean,
                std=self.std,
                io_retries=self.io_retries,
                io_retry_delay=self.io_retry_delay,
                io_resample_on_error=self.io_resample_on_error,
                io_resample_limit=self.io_resample_limit,
                skip_missing_pairs=self.skip_missing_pairs,
            )
            self.val_dataset = AOIMaskSegmentationDataset(
                images_dir=self.val_images,
                masks_dir=self.val_masks,
                tile_size=self.tile_size,
                mean=self.mean,
                std=self.std,
                io_retries=self.io_retries,
                io_retry_delay=self.io_retry_delay,
                io_resample_on_error=self.io_resample_on_error,
                io_resample_limit=self.io_resample_limit,
                skip_missing_pairs=self.skip_missing_pairs,
            )

        if stage in (None, "predict") and self.predict_images:
            self.predict_dataset = InferenceImageDataset(
                images_dir=self.predict_images,
                tile_size=self.tile_size,
                mean=self.mean,
                std=self.std,
                io_retries=self.io_retries,
                io_retry_delay=self.io_retry_delay,
                io_resample_on_error=self.io_resample_on_error,
                io_resample_limit=self.io_resample_limit,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train_dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val_dataloader.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.predict_dataset is None:
            raise RuntimeError("Call setup('predict') before requesting predict_dataloader.")
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
