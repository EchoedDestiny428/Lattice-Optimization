import os
import json
import shutil
import numpy as np
from scipy.ndimage import label

from generator import generate_lattice, RESOLUTION, BOX_SIZE


TARGET_DENSITY = 0.3
NUM_SAMPLES = 10

DATASET_DIR = "data/samples"

if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

os.makedirs(DATASET_DIR)


SHAPES = ["gyroid"]


for i in range(NUM_SAMPLES):

    sample_id = f"sample_{i:06d}"
    sample_dir = os.path.join(DATASET_DIR, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    shape_type = np.random.choice(SHAPES)
    freq = np.random.uniform(0.8, 1.2)
    noise = np.random.uniform(0.0, 0.05)

    voxels, density, threshold = generate_lattice(
        shape_type=shape_type,
        target_density=TARGET_DENSITY,
        freq=freq,
        noise=noise,
    )

    # --------------------------------------------------------
    # STRICT 6-CONNECTIVITY CLEANUP (Fixes hinge/pivot errors)
    # --------------------------------------------------------
    # Create a 3D cross footprint (only counts face-sharing neighbors)
    structure_6 = np.zeros((3, 3, 3), dtype=int)
    structure_6[1, 1, :] = 1  # Z-axis
    structure_6[1, :, 1] = 1  # Y-axis
    structure_6[:, 1, 1] = 1  # X-axis

    # Label using the strict face-sharing footprint
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
        "density": density,
        "threshold": float(threshold),
        "freq": float(freq),
        "noise": float(noise),
        "resolution": RESOLUTION,
        "box_size": BOX_SIZE,
        "solid_voxels": int(voxels.sum()),
    }

    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(sample_id, voxels.shape, density)

    # Verify component isolation using the same strict metric
    labels_after, n_after = label(voxels, structure=structure_6)

    sizes_after = []
    for comp in range(1, n_after + 1):
        sizes_after.append((labels_after == comp).sum())

    print("components after cleanup:", n_after)
    print("sizes after cleanup:", sizes_after)