import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import trimesh
import numpy as np

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

class GLU3DDataset(Dataset):
    def __init__(self, h5_path, grid_size=64):
        self.h5_path = h5_path
        self.grid_size = grid_size
        with h5py.File(self.h5_path, 'r') as f:
            self.dataset_length = f['voxels'].shape[0]
        self.scale_factor = 1e7  

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            voxel_grid = f['voxels'][idx].astype(np.float32)
            stiffness_matrix = f['stiffness'][idx]
            true_score = stiffness_matrix[0, 0] / self.scale_factor 

        x = torch.tensor(voxel_grid).unsqueeze(0)
        y = torch.tensor(true_score, dtype=torch.float32).unsqueeze(0)
        return x, y

def stl_to_voxel_tensor(stl_path, grid_size=64):
    mesh = trimesh.load(stl_path)
    stl_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # Use the identical 0.0 to 1.0 bounding domain from your working verify.py
    bounds = np.linspace(0.0, 1.0, grid_size)
    x_coords, y_coords = np.meshgrid(bounds, bounds, indexing='ij')
    
    ray_origins = np.vstack((x_coords.ravel(), y_coords.ravel(), np.full_like(x_coords.ravel(), -0.1))).T
    ray_directions = np.tile([0, 0, 1], (len(ray_origins), 1))
    
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, _ = intersector.intersects_location(
        ray_origins=ray_origins, 
        ray_directions=ray_directions, 
        multiple_hits=True
    )
    
    for ray_idx in range(len(ray_origins)):
        hit_mask = (index_ray == ray_idx)
        if not np.any(hit_mask):
            continue
            
        z_hits = locations[hit_mask, 2]
        z_hits = np.sort(z_hits)
        
        x_pixel = int(round(ray_origins[ray_idx, 0] * (grid_size - 1)))
        y_pixel = int(round(ray_origins[ray_idx, 1] * (grid_size - 1)))
        
        for i in range(0, len(z_hits) - 1, 2):
            z_start = max(0, min(grid_size - 1, int(round(z_hits[i] * (grid_size - 1)))))
            z_end = max(0, min(grid_size - 1, int(round(z_hits[i+1] * (grid_size - 1)))))
            stl_matrix[x_pixel, y_pixel, z_start:z_end + 1] = 1.0

    print("\n--- PRODUCTION STL VOXELIZER ---")
    print(f"  * Solid Voxel Count: {int(np.sum(stl_matrix))}")
    print(f"  * Density Ratio:     {np.sum(stl_matrix) / stl_matrix.size:.4f}")
    print("-" * 40)
    
    # Returns a single-channel 4D tensor: [1, 64, 64, 64]
    return torch.tensor(stl_matrix).unsqueeze(0)


if __name__ == "__main__":
    GRID_RESOLUTION = 64
    BATCH_SIZE = 16
    EPOCHS = 15  
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing hardware: {device}")
    
    sample_h5 = os.path.join("data", "h5_files", "test.h5")
    sample_stl = os.path.join("data", "stl_files", "standardized_lattice.stl")
    
    if os.path.exists(sample_h5):
        print("\n--- PHASE 1: Splitting Dataset ---")
        full_dataset = GLU3DDataset(sample_h5, grid_size=GRID_RESOLUTION)
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Total Dataset Samples: {len(full_dataset)}")
        print(f"Training on:          {len(train_dataset)} samples")
        print(f"Validating on:        {len(val_dataset)} samples")
        
        print("\n--- PHASE 2: Training & Validating the 3D CNN ---")
        model = Lattice3DCNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(EPOCHS):
            model.train()
            running_train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item() * inputs.size(0)
            
            epoch_train_loss = running_train_loss / len(train_dataset)
            
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                    running_val_loss += loss.item() * inputs.size(0)
                    
            epoch_val_loss = running_val_loss / len(val_dataset)
            
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), "lattice_3dcnn_best.pth")
                print(f"--> New best model saved at Epoch {best_epoch:02d} with Val Loss: {best_val_loss:.4f}")
            
        print(f"\nTraining complete! Best weights were captured at Epoch {best_epoch:02d}.")
        
        print("\nLoading best checkpoint weights for verification testing...")
        model.load_state_dict(torch.load("lattice_3dcnn_best.pth"))
        model.eval().to(device)
        
        print("\n--- PHASE 4: Verification Tests ---")
        with h5py.File(sample_h5, 'r') as f:
            true_h5_val = f['stiffness'][0][0, 0]
        
        test_h5_input = full_dataset[0][0].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_h5 = model(test_h5_input)
            
        actual_pred_h5 = pred_h5.item() * full_dataset.scale_factor
        print(f"[H5 Index 0] Prediction: {actual_pred_h5:.4f} | Target True Value: {true_h5_val:.4f}")
        
        if os.path.exists(sample_stl):
            test_stl_input = stl_to_voxel_tensor(sample_stl, grid_size=GRID_RESOLUTION).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_stl = model(test_stl_input)
            actual_pred_stl = pred_stl.item() * full_dataset.scale_factor
            print(f"[STL File]    Prediction: {actual_pred_stl:.4f}")
            
    else:
        print(f"Could not find the dataset at '{sample_h5}' to begin training.")