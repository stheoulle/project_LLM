# Sprint 1 scaffold: basic image ingestion and training pipeline

This folder contains a minimal scaffold for Sprint 1: ingest images, build dataloaders, and run a baseline training pipeline using PyTorch Lightning.

Contents

- requirements.txt — minimal Python dependencies
- src/
  - data.py — Dataset and DataLoader helpers
  - model.py — baseline ResNet18-based model
  - train.py — PyTorch Lightning training script

Quick start

1. activate a virtual environment:
    conda activate pytorch

2. Install dependencies:
   pip install -r requirements.txt

3. Prepare data:
   - Ensure `data/raw/` contains images (or create a small CSV that points to a few sample images).
   - Example CSV columns: image_path,label

4. Run training (example):
   python src/train.py --train-csv data/tables/train.csv --val-csv data/tables/val.csv --root-dir data/raw --batch-size 16 --max-epochs 5

Notes

- The scaffold is intentionally minimal to get a baseline running quickly.
- After validating the pipeline, we can extend transforms, add augmentations, multi-input fusion (tabular + image), logging, and DVC/MLflow integration.
