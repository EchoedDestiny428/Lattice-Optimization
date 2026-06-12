import os
import torch
import torch.nn as nn
import trimesh
import numpy as np

# =====================================================================
# 1. perceptron
# =====================================================================
class LatticePredictor(nn.Module):
    def __init__(self, input_size):
        super(LatticePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

# =====================================================================
# 2. flatten to voxxel
# =====================================================================
def stl_to_voxel_vector(stl_path, grid_size=32):
    mesh = trimesh.load(stl_path)
    voxels = mesh.voxelized(pitch=mesh.extents.max() / grid_size)
    
    voxel_matrix = voxels.matrix.astype(np.float32)
    
    padded_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    min_x, min_y, min_z = map(min, zip(voxel_matrix.shape, (grid_size, grid_size, grid_size)))
    padded_matrix[:min_x, :min_y, :min_z] = voxel_matrix[:min_x, :min_y, :min_z]
    
    return padded_matrix.flatten()

# =====================================================================
# 3. exec
# =====================================================================
if __name__ == "__main__":
    GRID_RESOLUTION = 32
    INPUT_FEATURES = GRID_RESOLUTION ** 3 # 32^3=32,768
    
    print("initializing AI models...")
    model = LatticePredictor(input_size=INPUT_FEATURES)
    
    sample_stl = "data\\stl_files\\sample_lattice.stl"
    
    if os.path.exists(sample_stl):
        print(f"processing {sample_stl}...")
        input_data = stl_to_voxel_vector(sample_stl, grid_size=GRID_RESOLUTION)
        input_tensor = torch.tensor(input_data).unsqueeze(0) # Add batch dimension
        
        with torch.no_grad():
            predicted_score = model(input_tensor)
            
        print(f"success -> pred. score: {predicted_score.item():.4f}")
    else:
        print(f"place the stl file at '{sample_stl}' to test pipeline.")