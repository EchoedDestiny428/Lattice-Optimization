import os
import h5py
import trimesh
import numpy as np

h5_path = os.path.join("data", "h5_files", "test.h5")
output_stl_path = "h5_index0_true_lattice.stl"

if os.path.exists(h5_path):
    print(f"Reading dataset from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        voxel_grid = f['voxels'][0]
        true_stiffness = f['stiffness'][0][0, 0]
        
    print(f"--> Found Voxel Grid Shape: {voxel_grid.shape}")
    print(f"--> True Mechanical Stiffness: {true_stiffness:,.4f} Pa")
    
    print("\nConverting voxel matrix back into a 3D Mesh...")
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_grid)
    
    mesh.export(output_stl_path)
    
    print("-" * 50)
    print(f"SUCCESS! File saved to your directory as: '{output_stl_path}'")
    print("Open this file in Windows 3D Viewer or your CAD software to inspect it.")
    print("-" * 50)
else:
    print(f"Could not find the dataset file at {h5_path}")