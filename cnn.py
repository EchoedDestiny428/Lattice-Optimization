import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

# Import the bridge and model you created
from dataset import VoxelLatticeDataset
from model import LatticeCNN3D

# ============================================================
# 1. SETUP HARDWARE & CONFIGURATIONS
# ============================================================
CSV_PATH = "data/dataset_compiled.csv"
SAMPLES_DIR = "data/samples"
BATCH_SIZE = 64
EPOCHS = 120  # Bumped slightly to 120 to give augmented variations time to settle
LEARNING_RATE = 1e-5  # Locked at stable 1e-5 for 3D spatial stability

# Automatically leverage your RTX 4060 GPU via CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using execution device: {device}")

# ============================================================
# 2. PREPARE THE DATA SPLITS (Using dataset.py)
# ============================================================
print("\n[Stage 1/4] Loading and partitioning spatial tensor dataset...")
full_dataset = VoxelLatticeDataset(csv_path=CSV_PATH, samples_dir=SAMPLES_DIR)

# Splitting strategy: 80% Train, 20% Validation/Test evaluation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

# DataLoaders stream batches seamlessly to the neural network
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"-> Dataset Ready. Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ============================================================
# 3. INITIALIZE 3D CNN STRUCTURE (Using model.py)
# ============================================================
print("\n[Stage 2/4] Initializing 3D Convolutional Network layers...")
model = LatticeCNN3D().to(device)

# MSE Loss forces aggression on high-stiffness predictions
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================
# 4. TRAINING LOOP WITH PROGRESS TRACKING & AUGMENTATION
# ============================================================
print(f"\n[Stage 3/4] Running optimization updates over {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Progress bar linked directly across our active training mini-batches
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS:02d}", unit="batch")
    
    for batch_idx, (voxel_batch, target_batch) in enumerate(train_bar):
        # Push variables to your GPU
        voxel_batch = voxel_batch.to(device)
        target_batch = target_batch.to(device)
        
        # --------------------------------------------------------
        # 3D DATA AUGMENTATION (On-the-Fly Symmetries)
        # Tensor shape: [Batch, Channel, Depth, Height, Width]
        # --------------------------------------------------------
        # 50% chance to flip along Depth axis
        if np.random.rand() > 0.5:
            voxel_batch = torch.flip(voxel_batch, dims=[2])
        # 50% chance to flip along Height axis
        if np.random.rand() > 0.5:
            voxel_batch = torch.flip(voxel_batch, dims=[3])
        # 50% chance to flip along Width axis
        if np.random.rand() > 0.5:
            voxel_batch = torch.flip(voxel_batch, dims=[4])
            
        # Zero gradients out from the previous step
        optimizer.zero_grad()
        
        # Forward pass: Predict stiffness
        predictions = model(voxel_batch)
        loss = criterion(predictions, target_batch)
        
        # Backward pass: Compute physics gradients
        loss.backward()

        # Hard ceiling to prevent single-batch gradient explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        running_loss += loss.item() * voxel_batch.size(0)
        
        # Calculate real-time RUNNING average loss instead of instantaneous batch loss
        current_running_avg = running_loss / ((batch_idx + 1) * BATCH_SIZE)
        train_bar.set_postfix(Running_Avg_Loss=f"{current_running_avg:.4f}")
        
    epoch_loss = running_loss / len(train_dataset)
    print(f"   -> [Epoch {epoch+1:02d} Summary] True Training Loss Average: {epoch_loss:.4f}")
    
# ============================================================
# 5. VALIDATION ACCURACY SCORING (R² Metric)
# ============================================================
print("\n[Stage 4/4] Evaluating multi-topology predictive accuracy...")
model.eval()

all_targets = []
all_preds = []

with torch.no_grad():
    for voxel_batch, target_batch in val_loader:
        voxel_batch = voxel_batch.to(device)
        predictions = model(voxel_batch)
        
        # Explicitly detach and move to CPU before conversion to numpy
        all_targets.append(target_batch.detach().cpu().numpy())
        all_preds.append(predictions.detach().cpu().numpy())

# Flatten arrays to compute statistical standard metrics
y_true = np.vstack(all_targets).flatten()
y_pred = np.vstack(all_preds).flatten()

# Standard R2 metric formulation
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_score = 1 - (ss_res / (ss_tot + 1e-8))

print("\n==============================================")
print("          3D CNN VALIDATION METRICS          ")
print("==============================================")
print(f"R² Spatial Accuracy:   {r2_score * 100:.2f}%")
print("==============================================")

print("\nSample Comparisons (Actual vs 3D CNN Predicted):")
for i in range(min(10, len(y_true))):
    print(f"Actual: {y_true[i]:.4f} GPa | Predicted: {y_pred[i]:.4f} GPa")

torch.save(model.state_dict(), "lattice_cnn.pth")
print("\n-> Success! Model weights saved completely to 'lattice_cnn.pth'")