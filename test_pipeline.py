# test_pipeline.py
import torch
from torch.utils.data import Dataset, DataLoader
from model import UNet, DiceBCELoss, calculate_metrics

class DummyDataset(Dataset):
    def __init__(self, num_samples=20, size=128):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.rand(1, self.size, self.size)
        mask = (torch.rand(1, self.size, self.size) > 0.95).float()  # ~5% oil jaisa sparse mask
        return img, mask

device = torch.device("cpu")
dataset = DummyDataset(num_samples=20)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet(in_channels=1, num_classes=1).to(device)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Testing pipeline with dummy data...")
for epoch in range(2):
    model.train()
    total_loss = 0
    for images, masks in loader:
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

print("Pipeline test successful! Ready for real data.")