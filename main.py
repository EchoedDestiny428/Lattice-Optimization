import os
import h5py
import trimesh
import numpy as np

h5_path = os.path.join("data", "h5_files", "test.h5")
output_stl_path = "standardized_lattice.stl"

if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as f:
        voxel_grid = f['voxels'][0]
        
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_grid)
    
    normalization_scale = 1.0 / voxel_grid.shape[0] 
    mesh.apply_scale(normalization_scale)
    
    target_center = np.array([0.5, 0.5, 0.5])
    translation_vector = target_center - mesh.center_mass
    mesh.apply_translation(translation_vector)
    
    mesh.export(output_stl_path)