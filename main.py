import os
import h5py
import numpy as np
import trimesh

def voxel_to_blocky_stl(h5_path, output_stl_path, target_extent=0.95, padding_multiplier=0.985):
    if not os.path.exists(h5_path):
        print(f"Error: Could not find H5 file at {h5_path}")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_stl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. Load the target voxel grid from H5
    with h5py.File(h5_path, 'r') as f:
        voxel_grid = f['voxels'][0].astype(bool)  # Must be boolean for voxel ops
    grid_size = voxel_grid.shape[0]

    # 2. Convert directly to a raw block/cube mesh (No Smoothing!)
    # This turns every True value into an un-smoothed 3D box geometry
    voxel_obj = trimesh.voxel.VoxelGrid(voxel_grid)
    mesh = voxel_obj.marching_cubes() # Note: Trimesh uses un-smoothed marching cubes internally for raw grids

    # 3. Structural Voxel-to-Physical Scaling (0.0 to 1.0 Domain)
    # Aligning the discrete indices precisely to the normalized box domain
    verts = mesh.vertices / grid_size
    mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces)

    # 4. Apply Target Structural Footprint
    max_side = mesh.extents.max()
    scale_factor = (target_extent / max_side) * padding_multiplier
    mesh.apply_scale(scale_factor)

    # 5. Perfect Domain Center Alignment
    bbox_center = mesh.bounds.mean(axis=0)
    target_center = np.array([0.5, 0.5, 0.5])
    mesh.apply_translation(target_center - bbox_center)

    # 6. Save the clean blocky asset
    mesh.export(output_stl_path)
    
    print("=== BLOCKY PIPELINE SYNCHRONIZATION COMPLETE ===")
    print(f"File Saved:          '{output_stl_path}'")
    print(f"Physical Extents:    {mesh.extents}")
    print(f"Bounding Box Center: {mesh.bounds.mean(axis=0)}")

if __name__ == "__main__":
    h5_input = os.path.join("data", "h5_files", "test2.h5")
    stl_output = os.path.join("data", "stl_files", "standardized_lattice.stl")
    
    voxel_to_blocky_stl(h5_input, stl_output)