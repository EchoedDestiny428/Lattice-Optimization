import json
import numpy as np
import trimesh
import sys
from pathlib import Path

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATASET_DIR, STL_DIR, SAMPLES_DIR

def voxels_to_stl(num_samples):
    """
    Converts voxel arrays in DATASET_DIR to STL meshes in STL_DIR.
    """
    # Ensure output directory exists
    STL_DIR.mkdir(parents=True, exist_ok=True)
    converted_count = 0

    print(f"Reading from: {SAMPLES_DIR}")
    print(f"Writing to: {STL_DIR}")

    for i in range(num_samples):
        folder_name = f"sample_{i:06d}"
        sample_dir = SAMPLES_DIR / folder_name
        npz_path = sample_dir / "voxels.npz"
        
        if not npz_path.exists():
            continue
            
        # Metadata check for density naming
        json_path = sample_dir / "metadata.json"
        density_str = ""
        if json_path.exists():
            with open(json_path, 'r') as f:
                try:
                    metadata = json.load(f)
                    density = metadata.get("actual_density", 0.0)
                    density_str = f"_d{density:.3f}"
                except json.JSONDecodeError:
                    pass

        stl_path = STL_DIR / f"{folder_name}{density_str}.stl"

        # Idempotency: Skip if already exists
        if stl_path.exists():
            continue

        try:
            # 1. Load Voxel Data
            with np.load(npz_path) as data:
                voxel_grid = data['voxels']

            # Ensure binary format
            if voxel_grid.dtype != bool:
                voxel_grid = voxel_grid > 0

            if not np.any(voxel_grid):
                print(f"Warning: {folder_name} is empty. Skipping.")
                continue

            # 2. Convert to Mesh
            voxels = trimesh.voxel.VoxelGrid(voxel_grid)
            mesh = voxels.marching_cubes

            # 3. Integrity Check
            if not mesh.is_watertight:
                trimesh.repair.fill_holes(mesh)
                trimesh.repair.fix_normals(mesh)

            # 4. Export
            mesh.export(stl_path)
            converted_count += 1
            
            if i % 10 == 0:
                print(f"Processed: {folder_name}")
                
        except Exception as e:
            print(f"Failed {folder_name}: {e}")

    print(f"\nTransformation complete! {converted_count} new files created.")

if __name__ == "__main__":
    try:
        user_input = input("Enter number of samples to process: ")
        voxels_to_stl(int(user_input))
    except ValueError:
        print("Please enter a valid integer.")

if __name__ == "__main__":
    try:
        user_input = input("Enter number of samples to process: ")
        voxels_to_stl(int(user_input))
    except ValueError:
        print("Please enter a valid integer.")