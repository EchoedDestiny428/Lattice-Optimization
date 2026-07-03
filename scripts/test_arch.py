import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from config import CSV_PATH, SAMPLES_DIR, DEVICE, RESOLUTION
from src.dataset import VoxelLatticeDataset
import numpy as np
from sklearn.metrics import r2_score

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, 2, 1), # 48 -> 24
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, 2, 1), # 24 -> 12
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, 2, 1), # 12 -> 6
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, 3, 2, 1), # 6 -> 3
            nn.BatchNorm3d(128),
            nn.ReLU(True),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.mean(x, dim=[2, 3, 4]) # Global Average Pool -> 128
        return self.regressor(x)

def test_model():
    dataset = VoxelLatticeDataset(csv_path=CSV_PATH, samples_dir=SAMPLES_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = SimpleCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Added weight decay!
    
    for epoch in range(15): # 15 epochs is enough to see if it learns
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        # Eval
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds.append(model(x).cpu().numpy())
                targets.append(y.cpu().numpy())
                
        preds = np.vstack(preds).flatten()
        targets = np.vstack(targets).flatten()
        r2 = r2_score(targets, preds)
        print(f"Epoch {epoch+1}: Val R2 = {r2*100:.2f}%")

if __name__ == "__main__":
    test_model()
