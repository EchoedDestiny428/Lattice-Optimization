import os
import json
import shutil
import numpy as np

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
        noise=noise
    )

    np.savez_compressed(
        os.path.join(sample_dir, "voxels.npz"),
        voxels=voxels.astype(np.uint8)
    )

    metadata = {
        "sample_id": sample_id,
        "shape": shape_type,
        "density": density,
        "threshold": float(threshold),
        "freq": float(freq),
        "noise": float(noise),
        "resolution": RESOLUTION,
        "box_size": BOX_SIZE
    }

    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(sample_id, voxels.shape, density)