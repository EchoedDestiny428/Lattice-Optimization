import json
import os
import numpy as np
from generator import generate_lattice, BOX_SIZE

# ------------------
# SETTINGS
# ------------------
RESOLUTION = 32
TARGET_DENSITY = 0.3
amount = 10

DATASET_DIR = os.path.join("data", "samples")
os.makedirs(DATASET_DIR, exist_ok=True)


# ------------------
# DATA GENERATION LOOP
# ------------------
for i in range(amount):

    sample_dir = os.path.join(DATASET_DIR, f"sample_{i:06d}")
    os.makedirs(sample_dir, exist_ok=True)

    # ------------------
    # TPMS parameters (stable sampling)
    # ------------------
    c1 = np.random.uniform(0.5, 1.0) * np.random.choice([-1, 1])
    c2 = np.random.uniform(0.5, 1.0) * np.random.choice([-1, 1])
    c3 = np.random.uniform(0.5, 1.0) * np.random.choice([-1, 1])
    c4 = np.random.uniform(0.1, 0.6) * np.random.choice([-1, 1])
    c5 = np.random.uniform(0.05, 0.3) * np.random.choice([-1, 1])

    params = [c1, c2, c3, c4, c5]

    # ------------------
    # GENERATE LATTICE (VOXELS ONLY)
    # ------------------
    voxels, voxel_density, threshold = generate_lattice(
        params,
        target_density=TARGET_DENSITY
    )

    # ------------------
    # BASIC STATS
    # ------------------
    shape = voxels.shape
    occupancy = float(voxel_density)

    # ------------------
    # SAVE VOXELS (THIS IS YOUR FEM INPUT LATER)
    # ------------------
    np.savez_compressed(
        os.path.join(sample_dir, "voxels.npz"),
        voxels=voxels.astype(np.uint8)
    )

    # ------------------
    # METADATA
    # ------------------
    metadata = {
        "sample_id": f"sample_{i:06d}",

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
            "shape": list(shape),
            "box_size": BOX_SIZE
        },

        "voxelization": {
            "occupancy": occupancy
        }
    }

    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # ------------------
    # LABELS placeholder
    # ------------------
    labels = {
        "effective_modulus": None,
        "peak_force": None,
        "ea": None,
        "sea": None
    }

    with open(os.path.join(sample_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=4)

    # ------------------
    # DEBUG
    # ------------------
    print(
        f"{sample_dir}",
        shape,
        f"occupancy={occupancy:.3f}",
        "saved"
    )