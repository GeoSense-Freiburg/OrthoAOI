"""Standalone prediction script for AOI segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
from PIL import Image

import pytorch_lightning as pl
import yaml

from datamodule import AOIDataModule
from lightning_module import DINOv3AOISegmenter


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict AOI masks with a trained checkpoint")
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    config_args, _ = parser.parse_known_args()
    config = _load_config(config_args.config)

    parser.add_argument("--predict-images", default=config.get("predict_images"), help="Folder of images")
    parser.add_argument("--predict-output", default=config.get("predict_output", "predictions"), help="Output folder")
    parser.add_argument("--checkpoint", default=config.get("checkpoint", None), help="Path to Lightning checkpoint")
    parser.add_argument(
        "--checkpoint-best",
        action="store_true",
        help="Use checkpoint_dir/best.ckpt from config",
    )
    parser.add_argument(
        "--checkpoint-last",
        action="store_true",
        help="Use checkpoint_dir/last.ckpt from config",
    )

    parser.add_argument("--backbone", default=config.get("backbone", "dinov3_vits16"))
    parser.add_argument("--backbone-repo", default=config.get("backbone_repo", "facebookresearch/dinov3"))
    parser.add_argument("--backbone-source", default=config.get("backbone_source", "github"))
    parser.add_argument("--backbone-weights", default=config.get("backbone_weights"))

    parser.add_argument("--tile-size", type=int, default=config.get("tile_size", 518))
    parser.add_argument("--num-workers", type=int, default=config.get("num_workers", 4))
    parser.add_argument("--threshold", type=float, default=config.get("threshold", 0.5))
    parser.add_argument("--accelerator", default=config.get("accelerator", "auto"))
    parser.add_argument("--devices", type=int, default=config.get("devices", 1))
    parser.add_argument("--checkpoint-dir", default=config.get("checkpoint_dir", "checkpoint"))

    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=tuple(config.get("mean", (0.430, 0.411, 0.296))),
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=tuple(config.get("std", (0.213, 0.156, 0.143))),
    )

    return parser.parse_args()


def _flatten_predictions(predictions: Iterable[Any]) -> List[dict]:
    flattened: List[dict] = []
    for batch in predictions:
        if isinstance(batch, dict):
            flattened.append(batch)
        else:
            flattened.extend(batch)
    return flattened


def main() -> None:
    args = parse_args()
    if not args.predict_images:
        raise ValueError("Provide --predict-images (or set in config).")
    if args.checkpoint_best and args.checkpoint_last:
        raise ValueError("Use only one of --checkpoint-best or --checkpoint-last.")
    if args.checkpoint_best:
        args.checkpoint = str(Path(args.checkpoint_dir) / "best.ckpt")
    if args.checkpoint_last:
        args.checkpoint = str(Path(args.checkpoint_dir) / "last.ckpt")
    if not args.checkpoint:
        args.checkpoint = config.get("checkpoint")
    if not args.checkpoint:
        raise ValueError("Provide --checkpoint or use --checkpoint-best/--checkpoint-last.")

    datamodule = AOIDataModule(
        train_images=None,
        train_annotations=None,
        val_images=None,
        val_annotations=None,
        predict_images=args.predict_images,
        tile_size=args.tile_size,
        batch_size=1,
        num_workers=args.num_workers,
        mean=tuple(args.mean),
        std=tuple(args.std),
    )

    model = DINOv3AOISegmenter(
        backbone_name=args.backbone,
        backbone_repo=args.backbone_repo,
        backbone_source=args.backbone_source,
        backbone_weights=args.backbone_weights,
        lr_head=1e-3,
        lr_backbone=1e-4,
        weight_decay=1e-4,
        freeze_epochs=0,
        threshold=args.threshold,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=10,
    )

    predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=args.checkpoint)
    output_dir = Path(args.predict_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in _flatten_predictions(predictions):
        paths = item["paths"]
        masks = item["masks"]
        for path, mask in zip(paths, masks):
            mask_np = mask.squeeze().cpu().numpy()
            mask_img = Image.fromarray((mask_np > 0.5).astype(np.uint8) * 255, mode="L")
            output_path = output_dir / f"{Path(path).stem}_mask.png"
            mask_img.save(output_path)


if __name__ == "__main__":
    main()
