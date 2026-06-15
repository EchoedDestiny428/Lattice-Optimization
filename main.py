import os
import h5py
import numpy as np
import trimesh

def voxel_to_blocky_stl(h5_path, output_stl_path, target_extent=1.0, padding_multiplier=1.0):
    if not os.path.exists(h5_path):
        print(f"Error: Could not find H5 file at {h5_path}")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_stl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. Load the target voxel grid from H5
    with h5py.File(h5_path, 'r') as f:
        voxel_grid = f['voxels'][0].astype(bool)
    grid_size = voxel_grid.shape[0]

    # 2. Extract active voxel coordinates
    # Find the exact (x, y, z) indices where voxel is True
    z_indices, y_indices, x_indices = np.where(voxel_grid)
    
    # Create an array of 3D center points for our cubes
    centers = np.column_stack((x_indices, y_indices, z_indices))

    # 3. Manually build clean, independent 3D boxes
    # This guarantees absolutely zero optimization glitches or sliver triangles
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh = trimesh.util.concatenate([box.copy().apply_translation(c) for c in centers])

    # 4. Structural Voxel-to-Physical Scaling (0.0 to 1.0 Domain)
    # Since boxes are centered at integer indices, we offset by 0.5 to align bounds perfectly
    verts = (mesh.vertices + 0.5) / grid_size
    mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces)

    # 5. Apply Target Structural Footprint
    max_side = mesh.extents.max()
    scale_factor = (target_extent / max_side) * padding_multiplier
    mesh.apply_scale(scale_factor)

    # 6. Perfect Domain Center Alignment
    bbox_center = mesh.bounds.mean(axis=0)
    target_center = np.array([0.5, 0.5, 0.5])
    mesh.apply_translation(target_center - bbox_center)

    # 7. Save the immaculate blocky asset
    mesh.export(output_stl_path)
    
    print("=== IMMACULATE BLOCKY PIPELINE COMPLETE ===")
    print(f"File Saved:          '{output_stl_path}'")
    print(f"Physical Extents:    {mesh.extents}")
    print(f"Bounding Box Center: {mesh.bounds.mean(axis=0)}")

if __name__ == "__main__":
    h5_input = os.path.join("data", "h5_files", "test2.h5")
    stl_output = os.path.join("data", "stl_files", "standardized_lattice.stl")
    
    voxel_to_blocky_stl(h5_input, stl_output)