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

# 2. Reconstruct the clean Cartesian matrix from the STL
mesh = trimesh.load(stl_path)
grid_size = 64
stl_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

scale_min = 0.5 - (0.95 / 2.0)
scale_max = 0.5 + (0.95 / 2.0)
step = 0.95 / grid_size
half_step = step / 2.0
space_coords = np.linspace(scale_min + half_step, scale_max - half_step, grid_size)

ray_origins = []
ray_mapping = []
for x_idx in range(grid_size):
    for y_idx in range(grid_size):
        ray_origins.append([space_coords[x_idx], space_coords[y_idx], -0.1])
        ray_mapping.append((x_idx, y_idx))
        
intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
locations, index_ray, _ = intersector.intersects_location(
    ray_origins=np.array(ray_origins), ray_directions=np.tile([0, 0, 1], (len(ray_origins), 1)), multiple_hits=True
)

for ray_idx in range(len(ray_origins)):
    hit_mask = (index_ray == ray_idx)
    if not np.any(hit_mask): continue
    z_hits = np.sort(locations[hit_mask, 2])
    x_pixel, y_pixel = ray_mapping[ray_idx]
    for i in range(0, len(z_hits) - 1, 2):
        z_start = max(0, min(grid_size - 1, int(round((z_hits[i] - (scale_min + half_step)) / step))))
        z_end = max(0, min(grid_size - 1, int(round((z_hits[i+1] - (scale_min + half_step)) / step))))
        stl_matrix[x_pixel, y_pixel, z_start:z_end + 1] = True

print("--- GEOMETRIC MATRIX BRUTE-FORCE MATCHING ---")
print(f"Target H5 Voxel Volume:  {np.sum(h5_grid)}")
print(f"Source STL Voxel Volume: {np.sum(stl_matrix)}")

# 3. Test all 48 possible 3D transpositions and flip combinations
permutations = [
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
]

found_match = False
for perm in permutations:
    transposed = np.transpose(stl_matrix, perm)
    for flip_x in [1, -1]:
        for flip_y in [1, -1]:
            for flip_z in [1, -1]:
                variant = transposed[::flip_x, ::flip_y, ::flip_z]
                
                # Check intersection volume match
                intersection = np.logical_and(h5_grid, variant)
                match_percentage = (np.sum(intersection) / np.sum(h5_grid)) * 100
                
                if match_percentage > 90.0:
                    print(f"\n[!] MATCH FOUND!")
                    print(f"  * Transpose Order: {perm}")
                    print(f"  * Slice Inversions: X={flip_x}, Y={flip_y}, Z={flip_z}")
                    print(f"  * Spatial Overlap: {match_percentage:.2f}%")
                    found_match = True
                    break
if not found_match:
    print("\nNo direct match found > 90%. Printing highest overlap instead:")
    # Fallback to show best attempt
    best_overlap = 0
    for perm in permutations:
        transposed = np.transpose(stl_matrix, perm)
        intersection = np.logical_and(h5_grid, transposed)
        overlap = (np.sum(intersection) / np.sum(h5_grid)) * 100
        if overlap > best_overlap: best_overlap = overlap
    print(f"  * Highest plain-transpose overlap: {best_overlap:.2f}%")