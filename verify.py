import os
import h5py
import trimesh
import numpy as np

h5_path = os.path.join("data", "h5_files", "test.h5")
custom_stl_path = os.path.join("data", "stl_files", "sample_lattice_2.stl")

print("=== DEEP GEOMETRY DIAGNOSTIC ===")

# 1. Read H5 true voxel metrics
if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as f:
        h5_voxels = f['voxels'][0]
    print(f"H5 Index 0 Total Grid Size: {h5_voxels.shape}")
    print(f"H5 Index 0 Solid Voxels:    {int(np.sum(h5_voxels))}")
    print(f"H5 Index 0 Density Ratio:   {np.sum(h5_voxels) / h5_voxels.size:.4f}")
else:
    print("H5 file not found.")

# 2. Read your custom STL voxel metrics
if os.path.exists(custom_stl_path):
    mesh = trimesh.load(custom_stl_path)
    voxels = mesh.voxelized(pitch=mesh.extents.max() / 64)
    stl_matrix = voxels.matrix.astype(np.float32)
    
    print(f"\nCustom STL Physical Dimensions: {mesh.extents}")
    print(f"Custom STL Solid Voxels:        {int(np.sum(stl_matrix))}")
    print(f"Custom STL Density Ratio:       {np.sum(stl_matrix) / (64**3):.4f}")
else:
    print(f"Custom STL not found at {custom_stl_path}")