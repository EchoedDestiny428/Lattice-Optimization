import os
import h5py
import torch
import torch.nn as nn
import trimesh
import numpy as np

# =====================================================================
# 1. 3D Convolutional Neural Network
# =====================================================================
class Lattice3DCNN(nn.Module):
    def __init__(self):
        super(Lattice3DCNN, self).__init__()
        
        # Convolutions scan the 3D space to detect structural features
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # Downsamples 64^3 to 32^3
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), # Downsamples 32^3 to 16^3
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # Downsamples 16^3 to 8^3
            nn.ReLU(),
            nn.Flatten() # Flattens the remaining 8x8x8 space into a single row
        )
        
        # Final layers to calculate the predicted stiffness number
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
# 2. FILE PARSERS (Sized for the 64x64x64 H5 resolution)
# =====================================================================
def stl_to_voxel_tensor(stl_path, grid_size=64):
    mesh = trimesh.load(stl_path)
    voxels = mesh.voxelized(pitch=mesh.extents.max() / grid_size)
    voxel_matrix = voxels.matrix.astype(np.float32)
    
    padded_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    min_x, min_y, min_z = map(min, zip(voxel_matrix.shape, (grid_size, grid_size, grid_size)))
    padded_matrix[:min_x, :min_y, :min_z] = voxel_matrix[:min_x, :min_y, :min_z]
    
    # Keep it 3D, just add the 1-channel dimension upfront: (1, 64, 64, 64)
    return torch.tensor(padded_matrix).unsqueeze(0)

def h5_to_voxel_tensor(h5_path, index, grid_size=64):
    with h5py.File(h5_path, 'r') as f:
        voxel_matrix = f['voxels'][index].astype(np.float32)
        
    if voxel_matrix.shape != (grid_size, grid_size, grid_size):
        padded_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        min_x, min_y, min_z = map(min, zip(voxel_matrix.shape, (grid_size, grid_size, grid_size)))
        padded_matrix[:min_x, :min_y, :min_z] = voxel_matrix[:min_x, :min_y, :min_z]
        return torch.tensor(padded_matrix).unsqueeze(0)
        
    return torch.tensor(voxel_matrix).unsqueeze(0)

# =====================================================================
# 3. EXECUTION
# =====================================================================
if __name__ == "__main__":
    GRID_RESOLUTION = 64
    
    print("initializing 3D CNN AI model...")
    model = Lattice3DCNN()
    model.eval() # Set to evaluation mode
    
    # -----------------------------------------------------------------
    # OPTION A: Processing an STL File
    # -----------------------------------------------------------------
    sample_stl = os.path.join("data", "stl_files", "sample_lattice.stl")
    if os.path.exists(sample_stl):
        print(f"\n[STL MODE] processing {sample_stl}...")
        input_tensor = stl_to_voxel_tensor(sample_stl, grid_size=GRID_RESOLUTION).unsqueeze(0)
        
        with torch.no_grad():
            predicted_score = model(input_tensor)
        print(f"success -> stl 3D CNN pred. score: {predicted_score.item():.4f}")
    else:
        print(f"\n[STL MODE] Skipping. No file at '{sample_stl}'")

    # -----------------------------------------------------------------
    # OPTION B: Processing an H5 Dataset Entry
    # -----------------------------------------------------------------
    sample_h5 = os.path.join("data", "h5_files", "test.h5")
    if os.path.exists(sample_h5):
        TARGET_INDEX = 0
        print(f"\n[H5 MODE] processing {sample_h5} at index {TARGET_INDEX}...")
        try:
            input_tensor_h5 = h5_to_voxel_tensor(sample_h5, index=TARGET_INDEX, grid_size=GRID_RESOLUTION).unsqueeze(0)
            
            with torch.no_grad():
                predicted_score_h5 = model(input_tensor_h5)
            print(f"success -> h5 3D CNN pred. score: {predicted_score_h5.item():.4f}")
        except Exception as e:
            print(f"Error parsing H5 file: {e}")
    else:
        print(f"\n[H5 MODE] Skipping. No file at '{sample_h5}'")