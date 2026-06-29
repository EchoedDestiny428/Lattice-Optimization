import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

# Updated imports to match structure
from src.dataset import VoxelLatticeDataset
from src.model import LatticeCNN3D
from config import CSV_PATH, SAMPLES_DIR, MODEL_WEIGHTS_PATH, DEVICE

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 120  
LEARNING_RATE = 1e-5  

print(f"Using execution device: {DEVICE}")

# 1. Initialize Dataset
# Fixed: Passed the paths from config.py to the class
full_dataset = VoxelLatticeDataset(csv_path=CSV_PATH, samples_dir=SAMPLES_DIR)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. Initialize Model
model = LatticeCNN3D().to(DEVICE)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Training Loop
print(f"\n[Stage 3/4] Running optimization over {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS:02d}", unit="batch")
    
    for voxel_batch, target_batch in train_bar:
        voxel_batch = voxel_batch.to(DEVICE)
        target_batch = target_batch.to(DEVICE)
        
        # Augmentation (X and Y flips only)
        if np.random.rand() > 0.5:
            voxel_batch = torch.flip(voxel_batch, dims=[2]) # Flip X
        if np.random.rand() > 0.5:
            voxel_batch = torch.flip(voxel_batch, dims=[3]) # Flip Y
            
        optimizer.zero_grad()
        predictions = model(voxel_batch)
        loss = criterion(predictions, target_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * voxel_batch.size(0)
        train_bar.set_postfix(Loss=f"{loss.item():.4f}")
        
    print(f" -> [Epoch {epoch+1:02d}] Avg Loss: {running_loss / len(train_dataset):.4f}")

# 4. Evaluation
print("\n[Stage 4/4] Validating accuracy...")
model.eval()
all_targets, all_preds = [], []

with torch.no_grad():
    for voxel_batch, target_batch in val_loader:
        voxel_batch = voxel_batch.to(DEVICE)
        predictions = model(voxel_batch)
        all_targets.append(target_batch.detach().cpu().numpy())
        all_preds.append(predictions.detach().cpu().numpy())

y_true = np.vstack(all_targets).flatten()
y_pred = np.vstack(all_preds).flatten()

# R2 Calculation
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_score = 1 - (ss_res / (ss_tot + 1e-8))

print(f"\nFinal R² Spatial Accuracy: {r2_score * 100:.2f}%")
torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
print(f"-> Model saved: {MODEL_WEIGHTS_PATH}")