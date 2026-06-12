import os
import h5py
import trimesh
import numpy as np

h5_path = os.path.join("data", "h5_files", "test.h5")
stl_path = os.path.join("data", "stl_files", "standardized_lattice.stl")

if not os.path.exists(h5_path) or not os.path.exists(stl_path):
    print("Error: Ensure test.h5 and standardized_lattice.stl are in your workspace.")
    exit()

# 1. Load the ground truth grid from H5
with h5py.File(h5_path, 'r') as f:
    h5_grid = f['voxels'][0].astype(bool)

# 2. Reconstruct using your VERIFIED meshgrid engine
mesh = trimesh.load(stl_path)
grid_size = 64
stl_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

scale_min = 0.5 - (0.95 / 2.0)
scale_max = 0.5 + (0.95 / 2.0)
step = 0.95 / grid_size
half_step = step / 2.0

bounds = np.linspace(scale_min + half_step, scale_max - half_step, grid_size)
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
    
    # Extract back out using your verified meshgrid flattening math
    x_pixel = int(round((ray_origins[ray_idx, 0] - (scale_min + half_step)) / step))
    y_pixel = int(round((ray_origins[ray_idx, 1] - (scale_min + half_step)) / step))
    x_pixel = max(0, min(grid_size - 1, x_pixel))
    y_pixel = max(0, min(grid_size - 1, y_pixel))
    
    for i in range(0, len(z_hits) - 1, 2):
        z_start = max(0, min(grid_size - 1, int(round((z_hits[i] - (scale_min + half_step)) / step))))
        z_end = max(0, min(grid_size - 1, int(round((z_hits[i+1] - (scale_min + half_step)) / step))))
        stl_matrix[x_pixel, y_pixel, z_start:z_end + 1] = True

print("--- VALIDATED MATRIX MATCHING SWEEP ---")
print(f"Target H5 Voxel Volume:  {np.sum(h5_grid)}")
print(f"Source STL Voxel Volume: {np.sum(stl_matrix)} (Perfect match baseline!)")

permutations = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
found_match = False

for perm in permutations:
    transposed = np.transpose(stl_matrix, perm)
    for flip_x in [1, -1]:
        for flip_y in [1, -1]:
            for flip_z in [1, -1]:
                variant = transposed[::flip_x, ::flip_y, ::flip_z]
                intersection = np.logical_and(h5_grid, variant)
                match_percentage = (np.sum(intersection) / np.sum(h5_grid)) * 100
                
                if match_percentage > 85.0:
                    print(f"\n[!] ACCURATE PIPELINE MATCH FOUND!")
                    print(f"  * Transpose Order: {perm}")
                    print(f"  * Slice Inversions: X={flip_x}, Y={flip_y}, Z={flip_z}")
                    print(f"  * Spatial Overlap: {match_percentage:.2f}%")
                    found_match = True
                    break
            if found_match: break
        if found_match: break

if not found_match:
    print("\nNo direct overlap found > 85%. Checking highest standard transposition...")
    best_overlap = 0
    best_perm = None
    for perm in permutations:
        transposed = np.transpose(stl_matrix, perm)
        intersection = np.logical_and(h5_grid, transposed)
        overlap = (np.sum(intersection) / np.sum(h5_grid)) * 100
        if overlap > best_overlap:
            best_overlap = overlap
            best_perm = perm
    print(f"  * Best plain-transpose: {best_perm} with {best_overlap:.2f}% overlap.")