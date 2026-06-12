import os
import h5py
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
# 2. flatten to voxxel (For STL Files)
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
# 3. parse h5 entry (For H5 Files)
# =====================================================================
def h5_to_voxel_vector(h5_path, index, grid_size=32):
    with h5py.File(h5_path, 'r') as f:
        voxel_matrix = f['voxels'][index].astype(np.float32)
        
    if voxel_matrix.shape != (grid_size, grid_size, grid_size):
        padded_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        min_x, min_y, min_z = map(min, zip(voxel_matrix.shape, (grid_size, grid_size, grid_size)))
        padded_matrix[:min_x, :min_y, :min_z] = voxel_matrix[:min_x, :min_y, :min_z]
        return padded_matrix.flatten()
        
    return voxel_matrix.flatten()

# =====================================================================
# 4. dual-format exec
# =====================================================================
if __name__ == "__main__":
    GRID_RESOLUTION = 32
    INPUT_FEATURES = GRID_RESOLUTION ** 3 # 32^3=32,768
    
    print("initializing AI models...")
    model = LatticePredictor(input_size=INPUT_FEATURES)
    
    # -----------------------------------------------------------------
    # 1. Processing an STL File
    # -----------------------------------------------------------------
    sample_stl = os.path.join("data", "stl_files", "sample_lattice.stl")
    
    if os.path.exists(sample_stl):
        print(f"\n[STL MODE] processing {sample_stl}...")
        input_data = stl_to_voxel_vector(sample_stl, grid_size=GRID_RESOLUTION)
        input_tensor = torch.tensor(input_data).unsqueeze(0)
        
        with torch.no_grad():
            predicted_score = model(input_tensor)
            
        print(f"success -> stl pred. score: {predicted_score.item():.4f}")
    else:
        print(f"\n[STL MODE] Skipping. No file at '{sample_stl}'")

    # -----------------------------------------------------------------
    # 2. Processing an H5 Dataset Entry
    # -----------------------------------------------------------------
    sample_h5 = os.path.join("data", "h5_files", "test.h5")
    
    if os.path.exists(sample_h5):
        TARGET_INDEX = 0  # Look at the first lattice entry in the H5 container
        print(f"\n[H5 MODE] processing {sample_h5} at index {TARGET_INDEX}...")
        
        try:
            input_data_h5 = h5_to_voxel_vector(sample_h5, index=TARGET_INDEX, grid_size=GRID_RESOLUTION)
            input_tensor_h5 = torch.tensor(input_data_h5).unsqueeze(0)
            
            with torch.no_grad():
                predicted_score_h5 = model(input_tensor_h5)
                
            print(f"success -> h5 pred. score: {predicted_score_h5.item():.4f}")
        except Exception as e:
            print(f"Error parsing H5 file: {e}")
    else:
        print(f"\n[H5 MODE] Skipping. No file at '{sample_h5}'")