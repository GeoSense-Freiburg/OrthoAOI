"""LightningModule for AOI segmentation using DINOv3 backbones."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import resolve_weights


class DINOv3AOISegmenter(pl.LightningModule):
    """Binary AOI segmentation with a DINOv3 backbone and a minimal head."""

    def __init__(
        self,
        backbone_name: str,
        backbone_repo: str,
        backbone_source: str,
        backbone_weights: str | None,
        lr_head: float,
        lr_backbone: float,
        weight_decay: float,
        freeze_epochs: int,
        threshold: float,
        models_dir: str = "models",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        weights_path = resolve_weights(backbone_weights, models_dir)
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        default_weights = models_path / f"{backbone_name}.pth"

        if weights_path is None and default_weights.exists():
            weights_path = str(default_weights)
            print(f"Using cached backbone weights: {weights_path}")

        if weights_path:
            self.backbone, loaded_by_hub = self._load_backbone(
                backbone_repo,
                backbone_name,
                backbone_source,
                pretrained=False,
                weights=weights_path,
            )
            if not loaded_by_hub:
                state = torch.load(weights_path, map_location="cpu")
                self.backbone.load_state_dict(state, strict=False)
        else:
            self.backbone, _ = self._load_backbone(
                backbone_repo,
                backbone_name,
                backbone_source,
                pretrained=True,
                weights=None,
            )
            torch.save(self.backbone.state_dict(), default_weights)
            print(f"Saved pretrained backbone weights to: {default_weights}")

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

    @staticmethod
    def _load_backbone(
        backbone_repo: str,
        backbone_name: str,
        backbone_source: str,
        pretrained: bool,
        weights: str | None,
    ) -> tuple[nn.Module, bool]:
        if weights is not None:
            try:
                return (
                    torch.hub.load(
                        backbone_repo,
                        backbone_name,
                        source=backbone_source,
                        weights=weights,
                        pretrained=pretrained,
                    ),
                    True,
                )
            except TypeError:
                try:
                    return (
                        torch.hub.load(
                            backbone_repo,
                            backbone_name,
                            source=backbone_source,
                            weights=weights,
                        ),
                        True,
                    )
                except TypeError:
                    pass
            except ImportError:
                pass
        try:
            return (
                torch.hub.load(
                    backbone_repo,
                    backbone_name,
                    source=backbone_source,
                    pretrained=pretrained,
                ),
                False,
            )
        except (TypeError, ImportError):
            try:
                return (
                    torch.hub.load(
                        backbone_repo,
                        backbone_name,
                        source=backbone_source,
                    ),
                    False,
                )
            except ImportError:
                pass

        if "dinov3" in backbone_repo and backbone_name.startswith("dinov3_"):
            return (
                DINOv3AOISegmenter._load_dinov3_backbone_direct(
                    backbone_repo=backbone_repo,
                    backbone_name=backbone_name,
                    backbone_source=backbone_source,
                    pretrained=pretrained,
                    weights=weights,
                ),
                True,
            )

        raise RuntimeError(
            f"Unable to load backbone '{backbone_name}' from '{backbone_repo}'"
        )

    @staticmethod
    def _load_dinov3_backbone_direct(
        backbone_repo: str,
        backbone_name: str,
        backbone_source: str,
        pretrained: bool,
        weights: str | None,
    ) -> nn.Module:
        if backbone_source == "local":
            repo_path = Path(backbone_repo)
        else:
            repo_path = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"

        if not repo_path.exists():
            raise FileNotFoundError(f"DINOv3 repo cache not found: {repo_path}")

        sys.path.insert(0, str(repo_path))
        try:
            backbones_module = importlib.import_module("dinov3.hub.backbones")
            constructor = getattr(backbones_module, backbone_name)
            kwargs: dict[str, Any] = {"pretrained": pretrained}
            if weights is not None:
                kwargs["weights"] = weights
            return constructor(**kwargs)
        finally:
            if sys.path and sys.path[0] == str(repo_path):
                sys.path.pop(0)

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
