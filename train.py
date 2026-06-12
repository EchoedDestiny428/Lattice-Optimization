import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import trimesh
import numpy as np

# =====================================================================
# 1. AI BRAIN (3D Convolutional Neural Network)
# =====================================================================
class Lattice3DCNN(nn.Module):
    def __init__(self):
        super(Lattice3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten() 
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 8 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# =====================================================================
# 2. BATCH DATA LOADER (With Target Normalization)
# =====================================================================
class GLU3DDataset(Dataset):
    def __init__(self, h5_path, grid_size=64):
        self.h5_path = h5_path
        self.grid_size = grid_size
        with h5py.File(self.h5_path, 'r') as f:
            self.dataset_length = f['voxels'].shape[0]
        
        # This keeps our target numbers near small, manageable single digits
        self.scale_factor = 1e7  

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            voxel_grid = f['voxels'][idx].astype(np.float32)
            stiffness_matrix = f['stiffness'][idx]
            # Divide by scale factor to normalize
            true_score = stiffness_matrix[0, 0] / self.scale_factor 

        x = torch.tensor(voxel_grid).unsqueeze(0)
        y = torch.tensor(true_score, dtype=torch.float32).unsqueeze(0)
        return x, y

# =====================================================================
# 3. SINGLE FILE PARSERS (For Testing)
# =====================================================================
def stl_to_voxel_tensor(stl_path, grid_size=64):
    mesh = trimesh.load(stl_path)
    voxels = mesh.voxelized(pitch=mesh.extents.max() / grid_size)
    voxel_matrix = voxels.matrix.astype(np.float32)
    padded_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    min_x, min_y, min_z = map(min, zip(voxel_matrix.shape, (grid_size, grid_size, grid_size)))
    padded_matrix[:min_x, :min_y, :min_z] = voxel_matrix[:min_x, :min_y, :min_z]
    return torch.tensor(padded_matrix).unsqueeze(0)

# =====================================================================
# 4. RUN PIPELINE & TRAINING LOOP
# =====================================================================
if __name__ == "__main__":
    GRID_RESOLUTION = 64
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing hardware: {device}")
    
    sample_h5 = os.path.join("data", "h5_files", "test.h5")
    sample_stl = os.path.join("data", "stl_files", "sample_lattice.stl")
    
    if os.path.exists(sample_h5):
        print("\n--- PHASE 1: Training the 3D CNN (Normalized) ---")
        dataset = GLU3DDataset(sample_h5, grid_size=GRID_RESOLUTION)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = Lattice3DCNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Training Loss (Normalized MSE): {epoch_loss:.6f}")
            
        print("\nTraining complete! Model is now optimized.")
        
        # Switch model back to evaluation mode for testing
        model.eval().to("cpu")
        
        print("\n--- PHASE 2: Verification Tests ---")
        with h5py.File(sample_h5, 'r') as f:
            true_h5_val = f['stiffness'][0][0, 0]
        
        test_h5_input = dataset[0][0].unsqueeze(0)
        with torch.no_grad():
            pred_h5 = model(test_h5_input)
            
        # MULTIPLY BACK by dataset.scale_factor to see the real engineering values
        actual_pred_h5 = pred_h5.item() * dataset.scale_factor
        print(f"[H5 Index 0] Prediction: {actual_pred_h5:.4f} | Target True Value: {true_h5_val:.4f}")
        
        if os.path.exists(sample_stl):
            test_stl_input = stl_to_voxel_tensor(sample_stl, grid_size=GRID_RESOLUTION).unsqueeze(0)
            with torch.no_grad():
                pred_stl = model(test_stl_input)
            actual_pred_stl = pred_stl.item() * dataset.scale_factor
            print(f"[STL File]    Prediction: {actual_pred_stl:.4f}")
            
    else:
        print(f"Could not find the dataset at '{sample_h5}' to begin training.")

# torch.save(model.state_dict(), "lattice_3dcnn_model.pth")