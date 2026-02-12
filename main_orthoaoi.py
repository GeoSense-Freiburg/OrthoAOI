"""Entrypoint for training and inference of AOI segmentation."""

from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
from PIL import Image

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import AOIDataModule
from lightning_module import DINOv3AOISegmenter


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_weights(weights_value: str, models_dir: str) -> str:
    parsed = urllib.parse.urlparse(weights_value)
    if parsed.scheme in {"http", "https"}:
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        filename = Path(parsed.path).name or "dinov3_weights.pth"
        destination = models_path / filename
        if not destination.exists():
            urllib.request.urlretrieve(weights_value, destination)  # noqa: S310
        return str(destination)
    return weights_value


def _validate_args(args: argparse.Namespace) -> None:
    training_fields = [args.train_images, args.train_annotations, args.val_images, args.val_annotations]
    has_training = any(training_fields)
    if has_training and not all(training_fields):
        missing = []
        if not args.train_images:
            missing.append("train_images")
        if not args.train_annotations:
            missing.append("train_annotations")
        if not args.val_images:
            missing.append("val_images")
        if not args.val_annotations:
            missing.append("val_annotations")
        raise ValueError(f"Missing required training fields: {', '.join(missing)}")

    has_predict = bool(args.predict_images)
    if has_predict and not args.predict_output:
        raise ValueError("predict_output is required when predict_images is set.")

    if not has_training and not has_predict:
        raise ValueError("Provide either training fields or predict_images for inference.")

    if not args.backbone_weights:
        raise ValueError("Provide --backbone-weights (path or URL) or set it in the config file.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AOI segmentation with DINOv3 + Lightning")
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    config_args, _ = parser.parse_known_args()
    config = _load_config(config_args.config)

    parser.add_argument("--config", default=config_args.config, help="Path to a YAML config file")
    parser.add_argument("--train-images", default=config.get("train_images"), help="Directory with training image tiles")
    parser.add_argument("--train-annotations", default=config.get("train_annotations"), help="COCO JSON for training")
    parser.add_argument("--val-images", default=config.get("val_images"), help="Directory with validation image tiles")
    parser.add_argument("--val-annotations", default=config.get("val_annotations"), help="COCO JSON for validation")
    parser.add_argument("--predict-images", default=config.get("predict_images"), help="Optional folder of images for inference")
    parser.add_argument("--predict-output", default=config.get("predict_output", "predictions"), help="Output folder for predicted masks")

    parser.add_argument("--backbone", default=config.get("backbone", "dinov3_vits16"), help="DINOv3 model name")
    parser.add_argument("--backbone-repo", default=config.get("backbone_repo", "facebookresearch/dinov3"), help="torch.hub repo")
    parser.add_argument(
        "--backbone-source",
        default=config.get("backbone_source", "github"),
        choices=["github", "local"],
        help="torch.hub source: 'github' or 'local'",
    )
    parser.add_argument("--backbone-weights", default=config.get("backbone_weights"), help="Path or URL to DINOv3 weights")

    parser.add_argument("--epochs", type=int, default=config.get("epochs", 30))
    parser.add_argument("--freeze-epochs", type=int, default=config.get("freeze_epochs", 5))
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", 8))
    parser.add_argument("--tile-size", type=int, default=config.get("tile_size", 518))
    parser.add_argument("--lr-head", type=float, default=config.get("lr_head", 1e-3))
    parser.add_argument("--lr-backbone", type=float, default=config.get("lr_backbone", 1e-4))
    parser.add_argument("--weight-decay", type=float, default=config.get("weight_decay", 1e-4))
    parser.add_argument("--num-workers", type=int, default=config.get("num_workers", 4))
    parser.add_argument("--threshold", type=float, default=config.get("threshold", 0.5))
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--accelerator", default=config.get("accelerator", "auto"))
    parser.add_argument("--devices", type=int, default=config.get("devices", 1))

    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=tuple(config.get("mean", (0.430, 0.411, 0.296))),
        metavar=("M1", "M2", "M3"),
        help="Normalization mean",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=tuple(config.get("std", (0.213, 0.156, 0.143))),
        metavar=("S1", "S2", "S3"),
        help="Normalization std",
    )
    parser.add_argument("--checkpoint-dir", default=config.get("checkpoint_dir", "checkpoint"), help="Checkpoint output directory")
    parser.add_argument("--models-dir", default=config.get("models_dir", "models"), help="Local folder for backbone weights")

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
    pl.seed_everything(args.seed, workers=True)

    _validate_args(args)
    args.backbone_weights = _resolve_weights(args.backbone_weights, args.models_dir)

    datamodule = AOIDataModule(
        train_images=args.train_images,
        train_annotations=args.train_annotations,
        val_images=args.val_images,
        val_annotations=args.val_annotations,
        predict_images=args.predict_images,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mean=tuple(args.mean),
        std=tuple(args.std),
    )

    model = DINOv3AOISegmenter(
        backbone_name=args.backbone,
        backbone_repo=args.backbone_repo,
        backbone_source=args.backbone_source,
        backbone_weights=args.backbone_weights,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        freeze_epochs=args.freeze_epochs,
        threshold=args.threshold,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    if args.train_images and args.val_images:
        trainer.fit(model, datamodule=datamodule)
        if args.config:
            best_ckpt = str(Path(args.checkpoint_dir) / "best.ckpt")
            last_ckpt = str(Path(args.checkpoint_dir) / "last.ckpt")
            config = _load_config(args.config)
            config["checkpoint"] = best_ckpt
            config["checkpoint_last"] = last_ckpt
            with open(args.config, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)
    else:
        if not args.predict_images:
            raise ValueError("Provide train/val data or --predict-images for inference.")

    if args.predict_images:
        predictions = trainer.predict(model, datamodule=datamodule, ckpt_path="best")
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
