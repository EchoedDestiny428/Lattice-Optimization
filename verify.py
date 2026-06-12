import os
import h5py
import trimesh
import numpy as np

h5_path = os.path.join("data", "h5_files", "test.h5")
custom_stl_path = "standardized_lattice.stl"

print("=== DEEP GEOMETRY DIAGNOSTIC ===")

if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as f:
        h5_voxels = f['voxels'][0]
    print(f"H5 Index 0 Total Grid Size: {h5_voxels.shape}")
    print(f"H5 Index 0 Solid Voxels:    {int(np.sum(h5_voxels))}")
    print(f"H5 Index 0 Density Ratio:   {np.sum(h5_voxels) / h5_voxels.size:.4f}")
else:
    print("H5 file not found.")

if os.path.exists(custom_stl_path):
    mesh = trimesh.load(custom_stl_path)
    
    grid_size = 64
    lin_space = np.linspace(0.0, 1.0, grid_size)
    x_grid, y_grid, z_grid = np.meshgrid(lin_space, lin_space, lin_space, indexing='ij')
    grid_points = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).T
    
    contains = mesh.contains(grid_points)
    stl_matrix = contains.reshape((grid_size, grid_size, grid_size)).astype(np.float32)
    
    print(f"\nCustom STL Physical Dimensions: {mesh.extents}")
    print(f"Custom STL Solid Voxels:        {int(np.sum(stl_matrix))}")
    print(f"Custom STL Density Ratio:       {np.sum(stl_matrix) / (grid_size**3):.4f}")
else:
    print(f"Custom STL not found at {custom_stl_path}")