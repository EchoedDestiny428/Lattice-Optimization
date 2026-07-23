import json
import numpy as np
import trimesh
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SAMPLES_DIR

def voxels_to_stl():
    """
    Converts all voxel arrays in SAMPLES_DIR to STL meshes, 
    saving them directly inside their respective sample folders.
    """
    if not SAMPLES_DIR.exists():
        print(f"Error: {SAMPLES_DIR} does not exist.")
        return

    # Get all subdirectories (samples)
    samples = sorted([d for d in SAMPLES_DIR.iterdir() if d.is_dir()])
    
    print(f"Found {len(samples)} samples to process.")
    converted_count = 0

    for sample_dir in samples:
        npz_path = sample_dir / "voxels.npz"
        stl_path = sample_dir / "mesh.stl"

        if not npz_path.exists():
            continue

        # Idempotency: Skip if STL already exists
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
                print(f"Warning: {sample_dir.name} is empty. Skipping.")
                continue

            # 2. Convert to Mesh
            voxels = trimesh.voxel.VoxelGrid(voxel_grid)
            mesh = voxels.marching_cubes

            # 3. Integrity Check
            if not mesh.is_watertight:
                trimesh.repair.fill_holes(mesh)
                trimesh.repair.fix_normals(mesh)

            # 4. Export to the sample directory
            mesh.export(stl_path)
            converted_count += 1
            
            # Print progress every 10 items
            if converted_count % 10 == 0:
                print(f"Processed: {sample_dir.name}")
                
        except Exception as e:
            print(f"Failed {sample_dir.name}: {e}")

    print(f"\nTransformation complete! {converted_count} new STL files created.")

if __name__ == "__main__":
    voxels_to_stl()