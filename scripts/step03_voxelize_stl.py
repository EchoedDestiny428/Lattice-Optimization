import numpy as np
from src.voxelizer import voxelize_stl
from config import SAMPLES_DIR, RESOLUTION, BOX_SIZE_MM

def run_voxelization():
    # Gather all sample directories
    samples = sorted([d for d in SAMPLES_DIR.iterdir() if d.is_dir()])
    
    print(f"Starting voxelization for {len(samples)} samples...")
    
    for sample_dir in samples:
        sample_id = sample_dir.name
        # Look for the STL file inside the specific sample directory
        stl_path = sample_dir / "mesh.stl"
        output_path = sample_dir / "voxels_input.npz"

        if not stl_path.exists():
            print(f"Skipping {sample_id}: 'mesh.stl' not found in {sample_dir}.")
            continue

        try:
            # Use the generalized function from src.voxelizer
            voxels, density = voxelize_stl(stl_path, RESOLUTION, BOX_SIZE_MM)
            
            # Save results back to the sample directory
            np.savez_compressed(output_path, voxels=voxels)
            print(f"Processed {sample_id} | Density: {density:.4f}")
            
        except Exception as e:
            print(f"Failed {sample_id}: {e}")

if __name__ == "__main__":
    run_voxelization()