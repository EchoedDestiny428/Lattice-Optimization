import os
import json
import shutil
import numpy as np
from scipy.ndimage import label

from generator import generate_lattice, RESOLUTION, BOX_SIZE

# --- CONFIGURATION SETTINGS ---
NUM_SAMPLES = 10
DATASET_DIR = "data/samples"

if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
os.makedirs(DATASET_DIR)

SHAPES = ["gyroid", "primitive", "diamond", "i_wp", "neovius"]

SHAPE_IDS = {
    "gyroid": 0,
    "primitive": 1,
    "diamond": 2,
    "i_wp": 3,
    "neovius": 4
}

for i in range(NUM_SAMPLES):
    sample_id = f"sample_{i:06d}"
    sample_dir = os.path.join(DATASET_DIR, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    shape_type = np.random.choice(SHAPES)
    freq = np.random.uniform(0.8, 1.2)
    
    noise = np.random.uniform(0.0, 0.12)
    
    target_density = np.random.uniform(0.15, 0.45)

    voxels, actual_density, threshold = generate_lattice(
        shape_type=shape_type,
        target_density=target_density,
        freq=freq,
        noise=noise,
    )

    # --------------------------------------------------------
    # STRICT 6-CONNECTIVITY CLEANUP (Fixes hinge/pivot errors)
    # --------------------------------------------------------
    structure_6 = np.zeros((3, 3, 3), dtype=int)
    structure_6[1, 1, :] = 1  # Z-axis
    structure_6[1, :, 1] = 1  # Y-axis
    structure_6[:, 1, 1] = 1  # X-axis

    labels, n = label(voxels, structure=structure_6)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = np.argmax(sizes)
    voxels = labels == largest
    # --------------------------------------------------------

    np.savez_compressed(
        os.path.join(sample_dir, "voxels.npz"), voxels=voxels.astype(np.uint8)
    )

    metadata = {
        "sample_id": sample_id,
        "shape": shape_type,
        "shape_id": SHAPE_IDS[shape_type],
        "density": actual_density,
        "threshold": float(threshold),
        "freq": float(freq),
        "noise": float(noise),
        "resolution": RESOLUTION,
        "box_size": BOX_SIZE,
        "solid_voxels": int(voxels.sum()),
    }

    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"{sample_id} | Shape: {shape_type:<10} | Actual Density: {actual_density:.4f}")

    labels_after, n_after = label(voxels, structure=structure_6)

    sizes_after = []
    for comp in range(1, n_after + 1):
        sizes_after.append((labels_after == comp).sum())

    print("components after cleanup:", n_after)
    print("sizes after cleanup:", sizes_after)
    print("-" * 50)