#!/usr/bin/env python3
"""Train a DINO backbone (v2/v3) with a segmentation head for AOI mask prediction."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class AOICOCOSegmentationDataset(Dataset):
    """COCO-style dataset returning orthomosaic image tiles and binary AOI masks."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        tile_size: int = 518,
        normalize: bool = True,
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

        image_transforms = [
            transforms.Resize((self.tile_size, self.tile_size), antialias=True),
            transforms.ToTensor(),
        ]
        if normalize:
            image_transforms.append(
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            )

        self.image_transform = transforms.Compose(image_transforms)
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


class MockDINOBackbone(nn.Module):
    """Small offline-friendly ViT-like backbone for smoke tests."""

    def __init__(self, patch_size: int = 14, embed_dim: int = 128) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.proj(x)
        _, _, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        assert tokens.shape[1] == h * w
        return {"x_norm_patchtokens": tokens}


class DINOAOISegmenter(nn.Module):
    def __init__(self, backbone_name: str = "dinov3_vitb14", backbone_repo: str = "facebookresearch/dinov3") -> None:
        super().__init__()
        self.backbone = self._load_backbone(backbone_name, backbone_repo)
        self.patch_size = getattr(self.backbone, "patch_size", 14)
        self.embed_dim = getattr(self.backbone, "embed_dim", None) or getattr(
            self.backbone, "num_features", 768
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    @staticmethod
    def _load_backbone(backbone_name: str, backbone_repo: str) -> nn.Module:
        if backbone_name == "mock_vit":
            return MockDINOBackbone()

        try:
            return torch.hub.load(backbone_repo, backbone_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load DINO backbone from torch.hub. "
                f"Check internet access and the repo/name combination (repo={backbone_repo}, model={backbone_name}), "
                "or use --backbone mock_vit for smoke tests."
            ) from exc

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def _to_feature_map(self, features: Any, input_hw: Tuple[int, int]) -> torch.Tensor:
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
        feature_map = self._to_feature_map(features, input_hw=(x.shape[2], x.shape[3]))
        logits = self.segmentation_head(feature_map)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    cardinality = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.bce_weight <= 1.0):
        raise ValueError("--bce-weight must be between 0.0 and 1.0")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.freeze_epochs < 0:
        raise ValueError("--freeze-epochs must be >= 0")
    if args.freeze_epochs > args.epochs:
        raise ValueError("--freeze-epochs cannot be greater than --epochs")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.tile_size < 14:
        raise ValueError("--tile-size must be >= 14 for DINO patch backbones")


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_ds = AOICOCOSegmentationDataset(
        images_dir=args.train_images,
        annotations_file=args.train_annotations,
        tile_size=args.tile_size,
    )
    val_ds = AOICOCOSegmentationDataset(
        images_dir=args.val_images,
        annotations_file=args.val_annotations,
        tile_size=args.tile_size,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train/val datasets must not be empty.")

    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    bce_weight: float = 0.5,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            bce = criterion(logits, masks)
            d_loss = dice_loss(logits, masks)
            loss = bce_weight * bce + (1.0 - bce_weight) * d_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    current_val_loss: float,
) -> float:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, checkpoint_dir / f"epoch_{epoch:03d}.pth")

    updated_best = min(best_val_loss, current_val_loss)
    if current_val_loss <= best_val_loss:
        payload["best_val_loss"] = updated_best
        torch.save(payload, checkpoint_dir / "best_model.pth")

    return updated_best


def predict_on_folder(
    model: nn.Module,
    image_dir: str | Path,
    output_dir: str | Path,
    tile_size: int,
    device: torch.device,
    threshold: float = 0.5,
) -> None:
    transform = transforms.Compose(
        [
            transforms.Resize((tile_size, tile_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model.eval()
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff") for p in image_dir.glob(ext)]
    )

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            mask = (probs > threshold).astype(np.uint8) * 255
            Image.fromarray(mask, mode="L").save(output_dir / f"{image_path.stem}_mask.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3/DINOv2 AOI segmentation model")
    parser.add_argument("--train-images", required=True, help="Directory with training image tiles")
    parser.add_argument("--train-annotations", required=True, help="COCO JSON for training")
    parser.add_argument("--val-images", required=True, help="Directory with validation image tiles")
    parser.add_argument("--val-annotations", required=True, help="COCO JSON for validation")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint output directory")
    parser.add_argument("--predict-images", default=None, help="Optional folder of new orthos for inference")
    parser.add_argument("--predict-output", default="predictions", help="Output folder for predicted masks")

    parser.add_argument("--backbone", default="dinov3_vitb14", help="Backbone model name from torch.hub repo")
    parser.add_argument("--backbone-repo", default="facebookresearch/dinov3", help="torch.hub repo, e.g. facebookresearch/dinov3 or facebookresearch/dinov2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=518)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader = build_dataloaders(args)

    model = DINOAOISegmenter(backbone_name=args.backbone, backbone_repo=args.backbone_repo).to(device)
    model.set_backbone_trainable(False)
    optimizer = torch.optim.AdamW(model.segmentation_head.parameters(), lr=args.lr_head)

    best_val_loss = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            model.set_backbone_trainable(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune)

        train_loss = run_epoch(model, train_loader, device, optimizer, bce_weight=args.bce_weight)
        val_loss = run_epoch(model, val_loader, device, optimizer=None, bce_weight=args.bce_weight)

        best_val_loss = save_checkpoint(
            checkpoint_dir,
            epoch,
            model,
            optimizer,
            best_val_loss=best_val_loss,
            current_val_loss=val_loss,
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| best_val_loss={best_val_loss:.4f}"
        )

    if args.predict_images:
        best_model_path = checkpoint_dir / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
        predict_on_folder(
            model,
            image_dir=args.predict_images,
            output_dir=args.predict_output,
            tile_size=args.tile_size,
            device=device,
        )
        print(f"Saved predicted AOI masks to: {args.predict_output}")


if __name__ == "__main__":
    main()
