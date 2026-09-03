"""
Data loader for Oil Spill Detection - with augmentation for robustness
on a small training subset.
"""

import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import os
import cv2
import random


class SpillDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=128, max_samples=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.filenames = sorted(os.listdir(mask_dir))

        if max_samples is not None:
            self.filenames = self.filenames[:max_samples]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        with rasterio.open(img_path) as src:
            img = src.read(1).astype(np.float32)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img_tensor, mask_tensor

    def _augment(self, img, mask):
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        if random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0, 1)

        return img, mask


if __name__ == "__main__":
    ds = SpillDataset("dataset/images", "dataset/masks", image_size=128, max_samples=5, augment=True)
    print(f"Dataset size: {len(ds)}")
    img, mask = ds[0]
    print("Image shape:", img.shape, "Mask shape:", mask.shape)