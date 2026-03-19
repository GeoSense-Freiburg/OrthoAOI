# OrthoAOI
This project uses self-supervised DINO models trained on existing Areas of Interest (AOIs) to learn spatial and semantic patterns in orthomosaics and automatically predict meaningful AOIs in new, unseen orthos. The goal is to scale AOI discovery across large UAV datasets with minimal manual annotation.

## DINOv3 AOI Segmentation (PyTorch Lightning)
Use `python main.py` to train with the default config in `configs/aoi_segmentation_train.yaml`.

### Project layout
- `main.py`: entrypoint for train and predict
- `script/`: model, data, and utility code
- `configs/`: training and prediction configs
- `models/`: pretrained DINOv3 checkpoint
- `data/splits/`: CSV manifests used by the default training config

### Features
- Loads a DINOv3 backbone with a lightweight segmentation head.
- Uses a local `.pth` checkpoint by default.
- Uses a minimal segmentation head (Conv + upsample).
- Trains from either CSV manifests (`image,mask`) or paired image/mask folders.
- Loads binary AOI masks directly from mask images.
- Trains with BCEWithLogitsLoss.
- Freezes the DINO backbone for warmup epochs, then unfreezes for fine-tuning.
- Writes runtime outputs under `/tmp/script`.

### Example config-driven run
Edit `configs/aoi_segmentation_train.yaml` with your dataset manifest paths and DINOv3 checkpoint path.

```yaml
data:
  train_images: data/splits/train.csv
  val_images: data/splits/val.csv
model:
  backbone_name: dinov3_vits16plus
  backbone_weights: /absolute/path/to/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

Then run:
```bash
python main.py
```

### Prediction
Edit `configs/aoi_segmentation_predict.yaml` or set `data.predict_images` and `predict.checkpoint_path`, then run:

```bash
python main.py configs/aoi_segmentation_predict.yaml predict
```

Predicted masks are written to `predict.output_dir`.

### Notes
- Install dependencies from `requirements.txt`.
- Local weight files must exist and be non-empty.
- Training defaults to CSV manifests in `data/splits/*.csv`, which is faster than recursively scanning image folders.
- Runtime logs and checkpoints are written under `/tmp/script` by default.
- Keep tile sizes aligned with the backbone patch size. For `dinov3_vits16plus`, use multiples of `16`.
