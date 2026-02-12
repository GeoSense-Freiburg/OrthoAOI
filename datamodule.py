"""LightningDataModule for AOI segmentation."""

from __future__ import annotations

from typing import Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.dataset import AOICOCOSegmentationDataset, InferenceImageDataset


class AOIDataModule(pl.LightningDataModule):
    """DataModule providing train/val/predict loaders."""

    def __init__(
        self,
        train_images: str | None,
        train_annotations: str | None,
        val_images: str | None,
        val_annotations: str | None,
        predict_images: str | None,
        tile_size: int,
        batch_size: int,
        num_workers: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.train_images = train_images
        self.train_annotations = train_annotations
        self.val_images = val_images
        self.val_annotations = val_annotations
        self.predict_images = predict_images
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std

        self.train_dataset: AOICOCOSegmentationDataset | None = None
        self.val_dataset: AOICOCOSegmentationDataset | None = None
        self.predict_dataset: InferenceImageDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if not all(
                [self.train_images, self.train_annotations, self.val_images, self.val_annotations]
            ):
                raise ValueError("Training requires train/val images and annotations.")
            self.train_dataset = AOICOCOSegmentationDataset(
                images_dir=self.train_images,
                annotations_file=self.train_annotations,
                tile_size=self.tile_size,
                mean=self.mean,
                std=self.std,
            )
            self.val_dataset = AOICOCOSegmentationDataset(
                images_dir=self.val_images,
                annotations_file=self.val_annotations,
                tile_size=self.tile_size,
                mean=self.mean,
                std=self.std,
            )

        if stage in (None, "predict") and self.predict_images:
            self.predict_dataset = InferenceImageDataset(
                images_dir=self.predict_images,
                tile_size=self.tile_size,
                mean=self.mean,
                std=self.std,
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
