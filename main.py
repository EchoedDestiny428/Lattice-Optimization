import json
import os
import shutil
import numpy as np

from generator import (
    generate_lattice,
    BOX_SIZE,
    RESOLUTION
)

# ------------------
# SETTINGS
# ------------------
TARGET_DENSITY = 0.30
NUM_SAMPLES = 10

DATASET_DIR = os.path.join("data", "samples")

# optional: start fresh each run
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

os.makedirs(DATASET_DIR, exist_ok=True)


# ------------------
# SAMPLE GENERATION
# ------------------
for i in range(NUM_SAMPLES):

    sample_id = f"sample_{i:06d}"

    sample_dir = os.path.join(
        DATASET_DIR,
        sample_id
    )

    os.makedirs(sample_dir, exist_ok=True)

    # ------------------
    # TPMS parameters
    # ------------------
    c1 = np.random.uniform(0.5, 1.0) * np.random.choice([-1, 1])
    c2 = np.random.uniform(0.5, 1.0) * np.random.choice([-1, 1])
    c3 = np.random.uniform(0.5, 1.0) * np.random.choice([-1, 1])

    c4 = np.random.uniform(0.1, 0.6) * np.random.choice([-1, 1])
    c5 = np.random.uniform(0.05, 0.3) * np.random.choice([-1, 1])

    params = [c1, c2, c3, c4, c5]

    # ------------------
    # Generate lattice
    # ------------------
    voxels, voxel_density, threshold = generate_lattice(
        params,
        target_density=TARGET_DENSITY
    )

    occupancy = float(voxel_density)

    # ------------------
    # Save voxel grid
    # ------------------
    np.savez_compressed(
        os.path.join(sample_dir, "voxels.npz"),
        voxels=voxels.astype(np.uint8)
    )

    # ------------------
    # Metadata
    # ------------------
    metadata = {
        "sample_id": sample_id,

        "generation": {
            "target_density": TARGET_DENSITY,
            "actual_density": occupancy,
            "threshold": float(threshold)
        },

        "parameters": {
            "c1": float(c1),
            "c2": float(c2),
            "c3": float(c3),
            "c4": float(c4),
            "c5": float(c5)
        },

        "grid": {
            "resolution": RESOLUTION,
            "shape": list(voxels.shape),
            "box_size": BOX_SIZE
        },

        "voxelization": {
            "occupancy": occupancy,
            "solid_voxels": int(voxels.sum())
        }
    }

    with open(
        os.path.join(sample_dir, "metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    # ------------------
    # Placeholder labels
    # ------------------
    labels = {
        "effective_modulus": None,
        "peak_force": None,
        "ea": None,
        "sea": None
    }

    with open(
        os.path.join(sample_dir, "labels.json"),
        "w"
    ) as f:
        json.dump(labels, f, indent=4)

    print(
        f"{sample_id}",
        f"shape={voxels.shape}",
        f"occupancy={occupancy:.3f}",
        f"solid_voxels={int(voxels.sum())}"
    )

print("\nDataset generation complete.")