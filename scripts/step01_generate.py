import os
import json
import shutil
import numpy as np
from scipy.ndimage import label, generate_binary_structure

# Import from config and src
from config import RESOLUTION, BOX_SIZE_MM, DATASET_DIR, NUM_SAMPLES, SAMPLES_DIR
from src.generator import (
    generate_harmonic_field, 
    generate_voronoi_field, 
    generate_strut_field, 
    generate_noise_field, 
    find_threshold
)

def generate_voxel_samples(num_samples=NUM_SAMPLES):
    # Clean folders
    if os.path.exists(SAMPLES_DIR):
        shutil.rmtree(SAMPLES_DIR)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # Create 6-connectivity structure for label()
    struct_6 = generate_binary_structure(rank=3, connectivity=1)

    print(f"Starting direct voxel generation of {NUM_SAMPLES} samples...")

    topology_types = ["Harmonic", "Voronoi", "Struts", "Noise"]

    for i in range(NUM_SAMPLES):
        sample_id = f"sample_{i:06d}"
        sample_dir = os.path.join(SAMPLES_DIR, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # 1. Procedural Parameters
        complexity = np.random.randint(2, 6) 
        target_density = np.random.uniform(0.2, 0.4) 
        topology = np.random.choice(topology_types)

        # 2. Generate Field based on topology
        if topology == "Harmonic":
            field = generate_harmonic_field(complexity=complexity)
        elif topology == "Voronoi":
            field = generate_voronoi_field(complexity=complexity)
        elif topology == "Struts":
            field = generate_strut_field(complexity=complexity)
        elif topology == "Noise":
            field = generate_noise_field(complexity=complexity)
        
        # 3. Solve for threshold
        threshold = find_threshold(field, target_density=target_density)
        
        # 4. Direct Voxelization
        voxels_matrix = (field < threshold).astype(np.uint8)

        # 5. Connectivity cleanup (Keep only largest component)
        labels_array, n = label(voxels_matrix, structure=struct_6)
        sizes = np.bincount(labels_array.ravel())
        sizes[0] = 0 # Ignore background
        
        if len(sizes) > 1:
            largest = np.argmax(sizes)
            voxels_matrix = (labels_array == largest).astype(np.uint8)
        
        # Calculate final stats
        actual_density = float(voxels_matrix.mean())

        # 6. Save
        np.savez_compressed(
            os.path.join(sample_dir, "voxels.npz"),
            voxels=voxels_matrix,
        )

        metadata = {
            "sample_id": sample_id,
            "topology": topology,
            "complexity": complexity,
            "target_density": target_density,
            "actual_density": actual_density,
            "threshold": float(threshold),
            "resolution": RESOLUTION,
            "box_size": BOX_SIZE_MM,
            "solid_voxels": int(voxels_matrix.sum()),
        }

        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        if i % 10 == 0:
            print(f"{sample_id} | Complexity={complexity} | Density={actual_density:.4f}")

    print("Generation Complete.")

if __name__ == "__main__":
    generate_voxel_samples()