# Configs

## orthoaoi.yaml
This file is read by `main_orthoaoi.py` when passed via `--config`.

Fields (all optional; CLI flags override config values):
- `train_images`: Path to training image tiles directory.
- `train_annotations`: Path to COCO JSON for training.
- `val_images`: Path to validation image tiles directory.
- `val_annotations`: Path to COCO JSON for validation.
- `predict_images`: Optional folder of images for inference.
- `predict_output`: Output folder for predicted masks.
- `backbone`: DINOv3 model name (e.g. `dinov3_vits16`).
- `backbone_repo`: `facebookresearch/dinov3`.
- `backbone_source`: `github` or `local` for torch.hub.
- `backbone_weights`: URL or local path to weights.
- `epochs`: Training epochs.
- `freeze_epochs`: Epochs to freeze backbone at the start.
- `batch_size`: Batch size.
- `tile_size`: Input tile size.
- `lr_head`: Learning rate for segmentation head.
- `lr_backbone`: Learning rate for backbone.
- `weight_decay`: Weight decay.
- `num_workers`: DataLoader workers.
- `threshold`: Sigmoid threshold for predicted masks.
- `seed`: Random seed.
- `accelerator`: Lightning accelerator (`auto`, `cpu`, `gpu`, etc.).
- `devices`: Number of devices.
- `mean`: Normalization mean (3 floats).
- `std`: Normalization std (3 floats).
- `checkpoint_dir`: Where Lightning saves checkpoints.
- `models_dir`: Where weights URLs are downloaded.
- `checkpoint`: (for prediction) Path to a Lightning checkpoint to load.
- `checkpoint_last`: Optional path to last checkpoint (auto-written after training).
- `checkpoint_dir`: Used by prediction helpers to resolve `best.ckpt` or `last.ckpt`.

## One-shot update helper
Use the helper script to populate or override config values:

```bash
python scripts/update_config.py \
  --train-images data/train/images \
  --train-annotations data/train/annotations.json \
  --val-images data/val/images \
  --val-annotations data/val/annotations.json \
  --backbone-weights https://<your-signed-url>
```
