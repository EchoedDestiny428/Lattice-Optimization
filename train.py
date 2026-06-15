import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import trimesh
import numpy as np

# 1. NETWORKING ARCHITECTURE
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
        return self.regressor(self.features(x))

# 2. RAW H5 DATA PIPELINE
class GLU3DDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
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

        return torch.tensor(voxel_grid).unsqueeze(0), torch.tensor(true_score).unsqueeze(0)

# 3. VERIFIED 0.0 TO 1.0 VOXELIZER
def stl_to_voxel_tensor(stl_path, grid_size=64):
    mesh = trimesh.load(stl_path)
    stl_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    bounds = np.linspace(0.0, 1.0, grid_size)
    x_coords, y_coords = np.meshgrid(bounds, bounds, indexing='ij')
    ray_origins = np.vstack((x_coords.ravel(), y_coords.ravel(), np.full_like(x_coords.ravel(), -0.1))).T
    ray_directions = np.tile([0, 0, 1], (len(ray_origins), 1))
    
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, _ = intersector.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True
    )
    
    for ray_idx in range(len(ray_origins)):
        hit_mask = (index_ray == ray_idx)
        if not np.any(hit_mask): continue
            
        z_hits = np.sort(locations[hit_mask, 2])
        x_pixel = int(round(ray_origins[ray_idx, 0] * (grid_size - 1)))
        y_pixel = int(round(ray_origins[ray_idx, 1] * (grid_size - 1)))
        
        for i in range(0, len(z_hits) - 1, 2):
            z_start = max(0, min(grid_size - 1, int(round(z_hits[i] * (grid_size - 1)))))
            z_end = max(0, min(grid_size - 1, int(round(z_hits[i+1] * (grid_size - 1)))))
            stl_matrix[x_pixel, y_pixel, z_start:z_end + 1] = 1.0

    return torch.tensor(stl_matrix).unsqueeze(0)

# 4. EXECUTION LOOP
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sample_h5 = os.path.join("data", "h5_files", "test.h5")
    sample_stl = os.path.join("data", "stl_files", "standardized_lattice.stl")
    
    # Setup final destination path
    model_dir = os.path.join("data", "models")
    os.makedirs(model_dir, exist_ok=True)
    final_weights_path = os.path.join(model_dir, "lattice_3dcnn_final.pth")
    
    # User Menu Selection
    print("=============================================")
    print("   Lattice 3D CNN Training Interface         ")
    print("=============================================")
    print("1) Load previously saved model weights")
    print("2) Train a brand new model from scratch")
    choice = input("\nSelect an option (1 or 2): ").strip()
    
    should_train = True
    model = Lattice3DCNN().to(device)
    
    if choice == '1':
        if os.path.exists(final_weights_path):
            print(f"\n[INFO] Found existing weights at: {final_weights_path}")
            model.load_state_dict(torch.load(final_weights_path, map_location=device))
            print("[SUCCESS] Pre-trained weights successfully loaded into model structure.")
            should_train = False
        else:
            print(f"\n[WARNING] Weights file not found at '{final_weights_path}'.")
            print("Defaulting back to training mode...")
    
    dataset = GLU3DDataset(sample_h5)
    
    # 5. OPTIONAL TRAINING BLOCK
    if should_train:
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("\nBeginning network training loop...")
        for epoch in range(15):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(dataset)
            print(f"Epoch [{epoch+1:02d}/15] Complete | Loss: {epoch_loss:.6f}")
            
        # ONLY SAVE HERE: Once training loop ends completely
        torch.save(model.state_dict(), final_weights_path)
        print(f"\n[SUCCESS] Training finished. Weights secured at: {final_weights_path}")
    
    # 6. VALIDATION TESTING RUN
    print("\nPreparing model evaluation pass...")
    model.eval()
    
    with torch.no_grad():
        with h5py.File(sample_h5, 'r') as f:
            true_h5_val = f['stiffness'][0][0, 0]
            
        h5_input = dataset[0][0].unsqueeze(0).to(device)
        pred_h5 = model(h5_input).item() * dataset.scale_factor
        print(f"\nH5 Target Baseline: {true_h5_val:.4f} | H5 Model Prediction: {pred_h5:.4f}")
        
        if os.path.exists(sample_stl):
            print(f"Voxelizing and running inference on: {sample_stl}")
            stl_input = stl_to_voxel_tensor(sample_stl).unsqueeze(0).to(device)
            pred_stl = model(stl_input).item() * dataset.scale_factor
            print(f"STL Model Prediction:  {pred_stl:.4f}")
        else:
            print(f"[ERROR] Verification STL missing at {sample_stl}. Skipping target check.")