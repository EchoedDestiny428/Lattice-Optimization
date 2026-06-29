import os
import json
import shutil
import numpy as np
import trimesh
from skimage import measure
from scipy.ndimage import label

# Import from your new structure
from config import RESOLUTION, BOX_SIZE_MM, DATASET_DIR, STL_DIR, NUM_SAMPLES
from src.generator import generate_random_lattice, find_threshold
from src.voxelizer import voxelize_stl



# Clean folders
for d in [DATASET_DIR, STL_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

print(f"Starting procedural generation of {NUM_SAMPLES} samples...")

for i in range(NUM_SAMPLES):
    sample_id = f"sample_{i:06d}"
    sample_dir = os.path.join(DATASET_DIR, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Procedural Parameters
    # Instead of specific shapes, we randomize complexity and density
    complexity = np.random.randint(2, 6) 
    target_density = np.random.uniform(0.15, 0.45)

    # 2. Generate Harmonic Field
    field = generate_random_lattice(complexity=complexity)
    
    # 3. Solve for threshold to hit specific density
    threshold = find_threshold(field, target_density=target_density)
    
    # Calculate actual density for metadata
    actual_density = float((field < threshold).mean())

    # 4. Marching Cubes (Mesh generation)
    verts, faces, _, _ = measure.marching_cubes(field, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.apply_scale(BOX_SIZE_MM / RESOLUTION)

    stl_path = os.path.join(STL_DIR, f"{sample_id}.stl")
    mesh.export(stl_path)

    # 5. Voxelize (The "Ground Truth" representation)
    voxels_matrix, _ = voxelize_stl(
        stl_path,
        resolution=RESOLUTION,
        box_size_mm=BOX_SIZE_MM,
    )

    # Connectivity cleanup (keep only largest component)
    structure_6 = np.zeros((3, 3, 3), dtype=int)
    structure_6[1, 1, :] = 1
    structure_6[1, :, 1] = 1
    structure_6[:, 1, 1] = 1

    labels_array, n = label(voxels_matrix, structure=structure_6)
    sizes = np.bincount(labels_array.ravel())
    sizes[0] = 0 # Ignore background
    largest = np.argmax(sizes)
    voxels_matrix = labels_array == largest

    # 6. Save
    np.savez_compressed(
        os.path.join(sample_dir, "voxels.npz"),
        voxels=voxels_matrix.astype(np.uint8),
    )

    metadata = {
        "sample_id": sample_id,
        "complexity": complexity,
        "target_density": target_density,
        "actual_density": actual_density,
        "threshold": float(threshold),
        "resolution": RESOLUTION,
        "box_size": BOX_SIZE_MM,
        "solid_voxels": int(voxels_matrix.sum()),
        "stl_path": stl_path,
    }

    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    if i % 10 == 0:
        print(f"{sample_id} | Complexity={complexity} | Density={actual_density:.4f}")

print("Generation Complete.")