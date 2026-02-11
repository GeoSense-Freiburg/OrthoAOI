# OrthoAOI
This project uses self-supervised DINO models trained on existing Areas of Interest (AOIs) to learn spatial and semantic patterns in orthomosaics and automatically predict meaningful AOIs in new, unseen orthos. The goal is to scale AOI discovery across large UAV datasets with minimal manual annotation.

## DINOv3 AOI Segmentation Training
Use `train_dinov2_aoi_segmentation.py` to train a DINO ViT backbone (DINOv3 by default, DINOv2 optional) with a lightweight segmentation head on COCO-style AOI labels.

### Features
- Loads pretrained DINOv3 backbones by default (for example `dinov3_vitb14` from `facebookresearch/dinov3`) via `torch.hub`.
- Supports switching back to DINOv2 by overriding `--backbone-repo facebookresearch/dinov2 --backbone dinov2_vitb14`.
- Includes `mock_vit` backbone mode for offline smoke tests and CI validation.
- Expects COCO-style image + annotation JSON for train/validation datasets.
- Builds binary AOI masks from polygon or RLE segmentation labels.
- Trains with a combined BCE + Dice loss.
- Freezes the DINO backbone for warmup epochs, then unfreezes for fine-tuning.
- Saves per-epoch checkpoints and a `best_model.pth` checkpoint.
- Optionally predicts AOI masks on new orthomosaic images after training.

### Example training command
```bash
python train_dinov2_aoi_segmentation.py \
  --train-images data/train/images \
  --train-annotations data/train/annotations.json \
  --val-images data/val/images \
  --val-annotations data/val/annotations.json \
  --checkpoint-dir checkpoints/dino_aoi \
  --backbone-repo facebookresearch/dinov3 \
  --backbone dinov3_vitb14 \
  --epochs 30 \
  --freeze-epochs 5 \
  --batch-size 8 \
  --tile-size 518 \
  --predict-images data/test/orthos \
  --predict-output outputs/test_masks
```

### Example smoke test (no network)
```bash
python train_dinov2_aoi_segmentation.py \
  --train-images tmp_demo/train_images \
  --train-annotations tmp_demo/train_annotations.json \
  --val-images tmp_demo/val_images \
  --val-annotations tmp_demo/val_annotations.json \
  --backbone mock_vit \
  --epochs 1 \
  --freeze-epochs 0 \
  --batch-size 1 \
  --tile-size 56 \
  --num-workers 0 \
  --device cpu
```

### Notes
- Install dependencies: `torch`, `torchvision`, `numpy`, `Pillow`, and `pycocotools` (for RLE labels).
- Keep tile sizes aligned with DINO patching (`14`-pixel patch size) for best results.
