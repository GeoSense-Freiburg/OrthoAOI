"""Simple entrypoint for OrthoAOI training/prediction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pytorch_lightning import Trainer

from script.data import AOIDataModule
from script.model import DINOv3AOISegmenter
from script.utils import save_prediction_masks

DEFAULT_CONFIG_PATH = "configs/aoi_segmentation_train.yaml"


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")
    return data


def _build_datamodule(config: dict[str, Any]) -> AOIDataModule:
    data_cfg = config.get("data", {})
    return AOIDataModule(
        train_images=data_cfg.get("train_images"),
        train_masks=data_cfg.get("train_masks"),
        val_images=data_cfg.get("val_images"),
        val_masks=data_cfg.get("val_masks"),
        predict_images=data_cfg.get("predict_images"),
        tile_size=data_cfg["tile_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        mean=tuple(data_cfg["mean"]),
        std=tuple(data_cfg["std"]),
        io_retries=data_cfg.get("io_retries", 3),
        io_retry_delay=data_cfg.get("io_retry_delay", 0.1),
        io_resample_on_error=data_cfg.get("io_resample_on_error", True),
        io_resample_limit=data_cfg.get("io_resample_limit", 10),
        skip_missing_pairs=data_cfg.get("skip_missing_pairs", True),
    )


def _build_trainer(config: dict[str, Any]) -> Trainer:
    return Trainer(**dict(config.get("trainer", {})))


def _resolve_config_path(argv: list[str]) -> str:
    if len(argv) < 2:
        return DEFAULT_CONFIG_PATH
    return argv[1]


def _resolve_mode(config: dict[str, Any], argv: list[str]) -> str:
    if len(argv) > 2:
        mode = argv[2].lower()
    else:
        mode = str(config.get("run", "fit")).lower()
    if mode not in {"fit", "predict", "test"}:
        raise SystemExit(f"Unsupported mode: {mode}. Use fit, predict, or test.")
    return mode


def _predict(config: dict[str, Any], datamodule: AOIDataModule) -> None:
    predict_cfg = config.get("predict", {})
    checkpoint_path = predict_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise SystemExit("Prediction requires predict.checkpoint_path in the config.")
    if not Path(checkpoint_path).exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not config.get("data", {}).get("predict_images"):
        raise SystemExit("Prediction requires data.predict_images in the config.")

    output_dir = predict_cfg.get("output_dir", "predictions")
    trainer = _build_trainer(config)
    model = DINOv3AOISegmenter.load_from_checkpoint(checkpoint_path)
    predictions = trainer.predict(model, datamodule=datamodule)
    save_prediction_masks(predictions, output_dir, threshold=model.hparams.threshold)


def main() -> None:
    config_path = _resolve_config_path(sys.argv)
    config = _load_config(config_path)
    mode = _resolve_mode(config, sys.argv)

    datamodule = _build_datamodule(config)

    if mode == "predict":
        _predict(config, datamodule)
        return

    model = DINOv3AOISegmenter(**config["model"])
    trainer = _build_trainer(config)
    trainer.fit(model, datamodule=datamodule)

    if mode == "test" or config.get("test"):
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
