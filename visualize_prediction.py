"""
Visualize model predictions - original image, predicted mask, and overlay.
Useful for demo screenshots and sanity-checking model output.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import rasterio
import cv2
from model import UNet


def visualize_result(image_path, model_path="unet_spill_checkpoint.pth", image_size=128, save_path="prediction_result.png"):
    device = torch.device("cpu")
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float32)
        orig_shape = img.shape

    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img_resized = cv2.resize(img_norm, (image_size, image_size))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = (torch.sigmoid(pred) > 0.5).float().squeeze().numpy()

    pred_mask_full = cv2.resize(pred_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_norm, cmap='gray')
    axes[0].set_title("Original SAR Image", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_mask_full, cmap='Reds')
    axes[1].set_title("Detected Oil Spill (Predicted Mask)", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(img_norm, cmap='gray')
    axes[2].imshow(pred_mask_full, cmap='Reds', alpha=0.45)
    axes[2].set_title("Overlay", fontsize=13, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")

    area_pixels = np.count_nonzero(pred_mask_full)
    print(f"Detected oil pixels: {area_pixels} ({100 * area_pixels / pred_mask_full.size:.3f}% of image)")


def visualize_ground_truth_comparison(image_path, mask_path, model_path="unet_spill_checkpoint.pth",
                                        image_size=128, save_path="prediction_vs_groundtruth.png"):
    """
    Side-by-side: original, ground truth mask, predicted mask.
    Useful for showing model accuracy during the demo/presentation.
    """
    device = torch.device("cpu")
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float32)
        orig_shape = img.shape

    with rasterio.open(mask_path) as src:
        gt_mask = src.read(1).astype(np.float32)

    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img_resized = cv2.resize(img_norm, (image_size, image_size))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = (torch.sigmoid(pred) > 0.5).float().squeeze().numpy()

    pred_mask_full = cv2.resize(pred_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    axes[0].imshow(img_norm, cmap='gray')
    axes[0].set_title("Original SAR Image", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='Greens')
    axes[1].set_title("Ground Truth Mask", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(pred_mask_full, cmap='Reds')
    axes[2].set_title("Model Prediction", fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(img_norm, cmap='gray')
    axes[3].imshow(gt_mask, cmap='Greens', alpha=0.4)
    axes[3].imshow(pred_mask_full, cmap='Reds', alpha=0.4)
    axes[3].set_title("Overlay (Green=GT, Red=Pred)", fontsize=13, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {save_path}")


if __name__ == "__main__":
    # After training completes, run this on a test image not used in training
    # Example (uncomment and adjust path once model + data are ready):
    # visualize_result("dataset/images/01199.tif")
    # visualize_ground_truth_comparison("dataset/images/01199.tif", "dataset/masks/01199.tif")
    print("Import and call visualize_result() or visualize_ground_truth_comparison() once training is complete.")