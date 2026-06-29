import numpy as np
from src.voxelizer import voxelize_stl  # Import the new module
from config import DATASET_DIR, SAMPLES_DIR, RESOLUTION, BOX_SIZE_MM
import os

STL_DIR = DATASET_DIR / "generated_stl"

def run_voxelization():
    samples = sorted([d for d in SAMPLES_DIR.iterdir() if d.is_dir()])
    
    print(f"Starting voxelization for {len(samples)} samples...")
    
    for sample_dir in samples:
        sample_id = sample_dir.name
        stl_path = STL_DIR / f"{sample_id}.stl"
        output_path = sample_dir / "voxels_input.npz"

        if not stl_path.exists():
            print(f"Skipping {sample_id}: STL not found.")
            continue

        try:
            # Use the generalized function
            voxels, density = voxelize_stl(stl_path, RESOLUTION, BOX_SIZE_MM)
            
            # Save results
            np.savez_compressed(output_path, voxels=voxels)
            print(f"Processed {sample_id} | Density: {density:.4f}")
            
        except Exception as e:
            print(f"Failed {sample_id}: {e}")

if __name__ == "__main__":
    run_voxelization()