import json
import os
import shutil
import numpy as np

from generator import generate_lattice, BOX_SIZE, RESOLUTION

# ------------------
# SETTINGS
# ------------------
TARGET_DENSITY = 0.30
NUM_SAMPLES = 10

DATASET_DIR = os.path.join("data", "samples")

# reset dataset
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

os.makedirs(DATASET_DIR, exist_ok=True)


# ------------------
# SHAPES
# ------------------
SHAPES = ["gyroid", "diamond", "primitive"]


# ------------------
# GENERATION LOOP
# ------------------
for i in range(NUM_SAMPLES):

    sample_id = f"sample_{i:06d}"
    sample_dir = os.path.join(DATASET_DIR, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    # pick shape
    shape_type = np.random.choice(SHAPES)

    # controlled variation
    freq = np.random.uniform(0.8, 1.3)
    noise = np.random.uniform(0.0, 0.08)

    # generate lattice
    voxels, voxel_density, threshold = generate_lattice(
        shape_type=shape_type,
        target_density=TARGET_DENSITY,
        freq=freq,
        noise=noise
    )

    occupancy = float(voxel_density)

    # save voxels
    np.savez_compressed(
        os.path.join(sample_dir, "voxels.npz"),
        voxels=voxels.astype(np.uint8)
    )

    # metadata
    metadata = {
        "sample_id": sample_id,

        "generation": {
            "shape_type": shape_type,
            "target_density": TARGET_DENSITY,
            "actual_density": occupancy,
            "threshold": float(threshold)
        },

        "variation": {
            "frequency": float(freq),
            "noise": float(noise)
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

    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # placeholder labels
    labels = {
        "effective_modulus": None,
        "peak_force": None,
        "ea": None,
        "sea": None
    }

    with open(os.path.join(sample_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=4)

    print(
        sample_id,
        shape_type,
        f"shape={voxels.shape}",
        f"occ={occupancy:.3f}"
    )

print("\nDataset generation complete.")