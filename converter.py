import os
import json
import numpy as np
import trimesh

def voxels_to_stl(num_samples):
    base_dir = os.path.join("data", "samples")
    output_dir = os.path.join("data", "sample_stl")
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    converted_count = 0

    for i in range(num_samples):
        folder_name = f"sample_{i:06d}"
        sample_dir = os.path.join(base_dir, folder_name)
        npz_path = os.path.join(sample_dir, "voxels.npz")
        json_path = os.path.join(sample_dir, "metadata.json")

        # Ensure the required source files exist
        if not os.path.exists(npz_path):
            print(f"Skipping: {npz_path} not found.")
            continue
        
        # Read density from metadata.json if it exists; fallback to a default if missing
        density_str = ""
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    density = metadata.get("density", 0.0)
                    # Format to exactly 3 decimal places
                    density_str = f"_density_{density:.3f}"
            except Exception as json_err:
                print(f"Warning: Could not read metadata for {folder_name}. Error: {json_err}")
        else:
            print(f"Warning: {json_path} missing. Proceeding without density in filename.")

        # Construct final output path with the appended density string
        stl_filename = f"{folder_name}{density_str}.stl"
        stl_path = os.path.join(output_dir, stl_filename)

        print(f"Processing {folder_name}...")

        try:
            # Load the npz file
            with np.load(npz_path) as data:
                key = data.files[0]
                voxel_grid = data[key]

            # Ensure the voxel data is boolean or binary
            if voxel_grid.dtype != bool:
                voxel_grid = voxel_grid > 0

            # Skip empty voxel grids
            if not np.any(voxel_grid):
                print(f"Skipping {folder_name}: Voxel grid is empty.")
                continue

            # Create a VoxelGrid object, then call marching_cubes
            voxels = trimesh.voxel.VoxelGrid(voxel_grid)
            mesh = voxels.marching_cubes

            # Export the mesh to STL format
            mesh.export(stl_path)
            print(f"Successfully saved: {stl_path}")
            converted_count += 1

        except Exception as e:
            print(f"Failed to process {folder_name}. Error: {e}")

    print(f"\nTransformation complete! Successfully converted {converted_count}/{num_samples} samples.")

if __name__ == "__main__":
    try:
        user_input = input("Enter the number of samples to convert (e.g., 100 for 0-99): ")
        count = int(user_input)
        if count <= 0:
            print("Please enter a number greater than 0.")
        else:
            voxels_to_stl(count)
    except ValueError:
        print("Invalid input. Please enter a valid integer.")