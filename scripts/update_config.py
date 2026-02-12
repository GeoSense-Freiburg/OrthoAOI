#!/usr/bin/env python3
"""Update configs/orthoaoi.yaml from CLI args."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _validate(config: dict) -> None:
    training_fields = ["train_images", "train_annotations", "val_images", "val_annotations"]
    has_training = any(config.get(field) for field in training_fields)
    if has_training and not all(config.get(field) for field in training_fields):
        missing = [field for field in training_fields if not config.get(field)]
        raise ValueError(f"Missing required training fields: {', '.join(missing)}")

    has_predict = bool(config.get("predict_images"))
    if has_predict and not config.get("predict_output"):
        raise ValueError("predict_output is required when predict_images is set.")

    if not has_training and not has_predict:
        raise ValueError("Provide either training fields or predict_images for inference.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update orthoaoi.yaml with CLI overrides")
    parser.add_argument("--config", default="configs/orthoaoi.yaml", help="Config path")
    parser.add_argument("--train-images")
    parser.add_argument("--train-annotations")
    parser.add_argument("--val-images")
    parser.add_argument("--val-annotations")
    parser.add_argument("--predict-images")
    parser.add_argument("--predict-output")
    parser.add_argument("--backbone")
    parser.add_argument("--backbone-repo")
    parser.add_argument("--backbone-source")
    parser.add_argument("--backbone-weights")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--freeze-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--lr-head", type=float)
    parser.add_argument("--lr-backbone", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--accelerator")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--mean", type=float, nargs=3)
    parser.add_argument("--std", type=float, nargs=3)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--models-dir")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    updates = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    config.update(updates)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    _validate(config)

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    main()
