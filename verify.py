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
    stl_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    bounds = np.linspace(0.0, 1.0, grid_size)
    x_coords, y_coords = np.meshgrid(bounds, bounds, indexing='ij')
    
    ray_origins = np.vstack((x_coords.ravel(), y_coords.ravel(), np.full_like(x_coords.ravel(), -0.1))).T
    ray_directions = np.tile([0, 0, 1], (len(ray_origins), 1))
    
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    
    # intersects_location returns: (3D hit locations, ray indices that hit, triangle indices hit)
    locations, index_ray, index_tri = intersector.intersects_location(
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
            stl_matrix[x_pixel, y_pixel, z_start:z_end + 1] = True

    stl_matrix = stl_matrix.astype(np.float32)
    
    print(f"\nCustom STL Physical Dimensions: {mesh.extents}")
    print(f"Custom STL Solid Voxels:        {int(np.sum(stl_matrix))}")
    print(f"Custom STL Density Ratio:       {np.sum(stl_matrix) / (grid_size**3):.4f}")
else:
    print(f"Custom STL not found at {custom_stl_path}")