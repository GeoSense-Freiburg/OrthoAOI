# OrthoAOI
This project uses self-supervised DINO models trained on existing Areas of Interest (AOIs) to learn spatial and semantic patterns in orthomosaics and automatically predict meaningful AOIs in new, unseen orthos. The goal is to scale AOI discovery across large UAV datasets with minimal manual annotation.

## DINOv3 AOI Segmentation (PyTorch Lightning)
Use `main_orthoaoi.py` to train a DINOv3 backbone with a lightweight segmentation head on COCO-style AOI labels.

### Features
- Loads pretrained DINOv3 backbones via `torch.hub` using official weights.
- Accepts a URL for weights; downloads them into `models/` automatically.
- Uses a minimal segmentation head (Conv + upsample).
- Expects COCO-style image + annotation JSON for train/validation datasets.
- Builds binary AOI masks from polygon or RLE segmentation labels.
- Trains with BCEWithLogitsLoss.
- Freezes the DINO backbone for warmup epochs, then unfreezes for fine-tuning.
- Uses Lightning checkpoints (best by `val_loss`).
- Optionally predicts AOI masks on new orthomosaic images after training.

### Example config-driven run
Edit `configs/orthoaoi.yaml` with your dataset paths and DINOv3 weights URL (from the access email).

```yaml
train_images: data/train/images
train_annotations: data/train/annotations.json
val_images: data/val/images
val_annotations: data/val/annotations.json
backbone_weights: https://<your-signed-url>
```

Then run:
```bash
python main_orthoaoi.py --config configs/orthoaoi.yaml
```

The weights URL will be downloaded into `models/` automatically.
After training, the config is updated with `checkpoint` and `checkpoint_last` paths.

### One-shot config helper
You can populate or update `configs/orthoaoi.yaml` from the CLI:

```bash
python scripts/update_config.py \
  --train-images data/train/images \
  --train-annotations data/train/annotations.json \
  --val-images data/val/images \
  --val-annotations data/val/annotations.json \
  --backbone-weights https://<your-signed-url>
```

### CLI override example
```bash
python main_orthoaoi.py \
  --config configs/orthoaoi.yaml \
  --epochs 10 \
  --batch-size 4 \
  --predict-images data/test/orthos \
  --predict-output outputs/test_masks
```

### Separate prediction script
```bash
python predict_orthoaoi.py \
  --config configs/orthoaoi.yaml \
  --checkpoint-best \
  --predict-images data/test/orthos \
  --predict-output outputs/test_masks
```

### Notes
- Install dependencies from `requirements.txt`.
- DINOv3 weights are accessed via signed URLs (time-limited). If downloads fail, request a fresh URL.
- Keep tile sizes aligned with DINO patching (`14`-pixel patch size) for best results.
