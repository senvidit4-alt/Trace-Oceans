import torch
from torch.utils.data import DataLoader, random_split
from data_loader import SpillDataset
from model import UNet, DiceBCELoss, calculate_metrics

# ---- Config ----
IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
IMAGE_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 30                 # upper cap - early stopping will likely finish sooner
LR = 1e-4
MAX_SAMPLES = 150
PATIENCE = 5                # stop if val IoU doesn't improve for this many epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Datasets (augmentation ONLY on training set, not validation) ----
full_dataset_no_aug = SpillDataset(IMAGE_DIR, MASK_DIR, image_size=IMAGE_SIZE, max_samples=MAX_SAMPLES, augment=False)
full_dataset_aug = SpillDataset(IMAGE_DIR, MASK_DIR, image_size=IMAGE_SIZE, max_samples=MAX_SAMPLES, augment=True)

train_size = int(0.8 * len(full_dataset_no_aug))
val_size = len(full_dataset_no_aug) - train_size

# Use the same split indices for both augmented/non-augmented versions
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = random_split(range(len(full_dataset_no_aug)), [train_size, val_size], generator=generator)

train_dataset = torch.utils.data.Subset(full_dataset_aug, train_indices.indices)
val_dataset = torch.utils.data.Subset(full_dataset_no_aug, val_indices.indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# ---- Model, Loss, Optimizer ----
model = UNet(in_channels=1, num_classes=1).to(device)
criterion = DiceBCELoss(bce_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- Training Loop with Early Stopping ----
best_iou = 0.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_iou, val_dice = 0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            iou, dice = calculate_metrics(preds, masks)
            val_iou += iou
            val_dice += dice

    avg_train_loss = train_loss / len(train_loader)
    avg_val_iou = val_iou / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val Dice: {avg_val_dice:.4f}")

    # ---- Early stopping + best model checkpoint ----
    if avg_val_iou > best_iou:
        best_iou = avg_val_iou
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "unet_spill_checkpoint.pth")
        print(f"  -> New best model saved (IoU: {best_iou:.4f})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered - no improvement for {PATIENCE} epochs.")
            break

print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
print("Best model saved as unet_spill_checkpoint.pth")