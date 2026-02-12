"""LightningModule for AOI segmentation using DINOv3 backbones."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3AOISegmenter(pl.LightningModule):
    """Binary AOI segmentation with a DINOv3 backbone and a minimal head."""

    def __init__(
        self,
        backbone_name: str,
        backbone_repo: str,
        backbone_source: str,
        backbone_weights: str,
        lr_head: float,
        lr_backbone: float,
        weight_decay: float,
        freeze_epochs: int,
        threshold: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = torch.hub.load(
            backbone_repo,
            backbone_name,
            source=backbone_source,
            weights=backbone_weights,
        )

        self.patch_size = self._get_patch_size(self.backbone)
        self.embed_dim = self._get_embed_dim(self.backbone)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self._backbone_frozen = False

    @staticmethod
    def _get_patch_size(backbone: nn.Module) -> int:
        if hasattr(backbone, "patch_size"):
            return int(backbone.patch_size)
        patch_embed = getattr(backbone, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "patch_size"):
            patch_size = patch_embed.patch_size
            return int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)
        return 16

    @staticmethod
    def _get_embed_dim(backbone: nn.Module) -> int:
        if hasattr(backbone, "embed_dim"):
            return int(backbone.embed_dim)
        if hasattr(backbone, "num_features"):
            return int(backbone.num_features)
        raise ValueError("Unable to infer DINOv3 embedding dimension.")

    def _set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable
        self._backbone_frozen = not trainable

    def on_fit_start(self) -> None:
        if self.hparams.freeze_epochs > 0:
            self._set_backbone_trainable(False)

    def on_train_epoch_start(self) -> None:
        if self.hparams.freeze_epochs > 0 and self.current_epoch == self.hparams.freeze_epochs:
            self._set_backbone_trainable(True)

    def _tokens_to_feature_map(
        self, features: Any, input_hw: Tuple[int, int]
    ) -> torch.Tensor:
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                tokens = features["x_norm_patchtokens"]
            elif "x_prenorm" in features:
                tokens = features["x_prenorm"][:, 1:]
            else:
                raise ValueError(f"Unsupported feature keys: {list(features.keys())}")
        else:
            tokens = features

        if tokens.ndim != 3:
            raise ValueError(f"Expected patch tokens [B, N, C], got {tokens.shape}")

        b, n, c = tokens.shape
        h = input_hw[0] // self.patch_size
        w = input_hw[1] // self.patch_size
        if h * w != n:
            side = int(math.sqrt(n))
            h, w = side, n // side
            if h * w != n:
                raise ValueError(f"Cannot infer feature map shape for {n} tokens")

        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        feature_map = self._tokens_to_feature_map(features, (x.shape[2], x.shape[3]))
        logits = self.segmentation_head(feature_map)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    def predict_step(
        self, batch: Tuple[torch.Tensor, str], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor | str]:
        images, paths = batch
        logits = self(images)
        probs = torch.sigmoid(logits)
        masks = (probs > self.hparams.threshold).float()
        return {"paths": paths, "masks": masks}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        head_params = list(self.segmentation_head.parameters())
        backbone_params = list(self.backbone.parameters())
        return torch.optim.AdamW(
            [
                {"params": head_params, "lr": self.hparams.lr_head},
                {"params": backbone_params, "lr": self.hparams.lr_backbone},
            ],
            weight_decay=self.hparams.weight_decay,
        )
