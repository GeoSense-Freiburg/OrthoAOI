# Configs

## aoi_segmentation_train.yaml
Default training config read by `python main.py`.

Top-level sections:
- `trainer`: Lightning trainer settings.
- `model`: DINOv3 backbone and optimization settings.
- `data`: DataModule settings. By default `train_images` and `val_images` point to CSV manifests in `data/splits/`.
- `test` (optional): Set to `true` to run `trainer.test` after training.

## aoi_segmentation_predict.yaml
Prediction config read by `python main.py configs/aoi_segmentation_predict.yaml predict`.

Required fields before prediction:
- `data.predict_images`: directory of images to segment
- `predict.checkpoint_path`: Lightning checkpoint to load
