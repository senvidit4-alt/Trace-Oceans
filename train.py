import torch
from torch.utils.data import DataLoader, random_split
from data_loader import SpillDataset
from model import UNet, DiceBCELoss, calculate_metrics

# ---- Config (CPU-friendly) ----
IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
IMAGE_SIZE = 128        # 256 se 128 kiya — CPU pe fast hoga
BATCH_SIZE = 2           # chhota batch size CPU ke liye
EPOCHS = 10              # kam epochs, demo ke liye kaafi hai
LR = 1e-4
MAX_SAMPLES = 150        # sirf 150 images use karenge, 1200 nahi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

full_dataset = SpillDataset(IMAGE_DIR, MASK_DIR, image_size=IMAGE_SIZE, max_samples=MAX_SAMPLES)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

model = UNet(in_channels=1, num_classes=1).to(device)
criterion = DiceBCELoss(bce_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

    torch.save(model.state_dict(), "unet_spill_checkpoint.pth")

print("Training complete. Model saved as unet_spill_checkpoint.pth")