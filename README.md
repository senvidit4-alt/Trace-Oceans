# Trace-Oceans: Oil Spill Detection with Deep Learning (U-Net)

An end-to-end semantic segmentation pipeline using PyTorch for detecting oil spills in satellite and aerial imagery.

## 📁 Directory Structure

```text
oil-spill-ml/
├── dataset/
│   ├── images/         # Satellite/aerial input images (.png, .jpg, .tif)
│   └── masks/          # Binary segmentation masks (.png, .jpg, .tif)
├── model.py            # U-Net architecture & loss metrics
├── data_loader.py      # Dataset loading & preprocessing
├── train.py            # Training and validation loop
├── inference.py        # Single image & batch inference with GeoTIFF support
├── check_dataset.py    # Dataset validation & statistics utility
├── inspect_mask.py     # Mask visualization & inspection utility
├── test_pipeline.py    # Pipeline integration test
└── requirements.txt    # Required Python packages
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Place your satellite images and corresponding binary masks into `dataset/images/` and `dataset/masks/`.
- Images & masks should have matching filenames (e.g. `sample_01.png` in images and `sample_01.png` in masks).
- Masks should be grayscale/binary images where white pixels (255) represent oil spills and black (0) represents clean water/land.

### 3. Run Training
```bash
python train.py --epochs 25 --batch-size 8 --lr 1e-4
```
*Tip: If the dataset folder is empty, the training script automatically creates synthetic ocean and oil spill samples so you can test immediately!*

### 4. Run Inference
```bash
python inference.py --image path/to/satellite_image.tif --checkpoint checkpoints/best_model.pth --output-dir ./predictions
```

### 5. Output Artifacts
- **Checkpoints**: Saved in `./checkpoints/best_model.pth` and `./checkpoints/last_model.pth`
- **Training Curves**: Saved as `training_curves.png`
- **Inference Results**: Segmentation mask overlay and geo-referenced outputs in `./predictions/`