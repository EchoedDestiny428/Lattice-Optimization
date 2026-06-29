import numpy as np
from tqdm import tqdm
from src.voxelizer import voxelize_stl
from config import SAMPLES_DIR, RESOLUTION, BOX_SIZE_MM

def run_voxelization():
    # Gather all sample directories
    samples = sorted([d for d in SAMPLES_DIR.iterdir() if d.is_dir()])
    
    # We wrap 'samples' in tqdm to create the progress bar
    # desc= sets the label, unit= sets the counter type
    progress_bar = tqdm(samples, desc="Voxelizing samples", unit="sample")
    
    for sample_dir in progress_bar:
        sample_id = sample_dir.name
        stl_path = sample_dir / "mesh.stl"
        output_path = sample_dir / "voxels_input.npz"

        # Check for STL
        if not stl_path.exists():
            tqdm.write(f"Skipping {sample_id}: 'mesh.stl' not found.")
            continue

        try:
            # Voxelize
            voxels, density = voxelize_stl(stl_path, RESOLUTION, BOX_SIZE_MM)
            
            # Save
            np.savez_compressed(output_path, voxels=voxels)
            
            # Update the description to show current status (optional)
            progress_bar.set_postfix({"id": sample_id, "d": f"{density:.3f}"})
            
        except Exception as e:
            tqdm.write(f"Failed {sample_id}: {e}")

if __name__ == "__main__":
    run_voxelization()