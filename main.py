import json
import os
import numpy as np
from generator import generate_lattice, BOX_SIZE

VOXEL_RESOLUTION = 64
TARGET_DENSITY = 0.3
amount = 10

DATASET_DIR = os.path.join("data", "samples")
os.makedirs(DATASET_DIR, exist_ok=True)


for i in range(amount):

    sample_dir = os.path.join(DATASET_DIR, f"sample_{i:06d}")
    os.makedirs(sample_dir, exist_ok=True)

    # ------------------
    # random TPMS params
    # ------------------
    c1 = np.random.uniform(-1.5, 1.5)
    c2 = np.random.uniform(-1.5, 1.5)
    c3 = np.random.uniform(-1.5, 1.5)
    c4 = np.random.uniform(-1.0, 1.0)
    c5 = np.random.uniform(-0.5, 0.5)

    # avoid degenerate field
    if abs(c1) + abs(c2) + abs(c3) + abs(c4) + abs(c5) < 0.2:
        c1 = 1.0

    params = [c1, c2, c3, c4, c5]

    # ------------------
    # generate lattice
    # ------------------
    mesh, voxels, voxel_density, thickness = generate_lattice(
        params,
        target_density=TARGET_DENSITY
    )

    # ------------------
    # mesh cleanup (safe for ANSYS)
    # ------------------
    mesh.process(validate=True)

    volume = float(mesh.volume)
    surface_area = float(mesh.area)


    actual_density = voxel_density

    # ------------------
    # Save STL
    # ------------------
    stl_path = os.path.join(sample_dir, "lattice.stl")
    mesh.export(stl_path)

    # ------------------
    # Save voxels
    # ------------------
    np.savez_compressed(
        os.path.join(sample_dir, "voxels.npz"),
        voxels=voxels
    )

    # ------------------
    # Metadata
    # ------------------
    metadata = {
        "sample_id": f"sample_{i:06d}",

        "generation": {
            "target_density": TARGET_DENSITY,
            "actual_density": float(actual_density),
            "thickness": float(thickness)
        },

        "parameters": {
            "c1": float(c1),
            "c2": float(c2),
            "c3": float(c3),
            "c4": float(c4),
            "c5": float(c5)
        },

        "geometry": {
            "volume": volume,
            "surface_area": surface_area,
        },

        "voxelization": {
            "resolution": VOXEL_RESOLUTION,
            "occupancy": float(voxel_density)
        },

        "box_size": BOX_SIZE
    }

    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # ------------------
    # Labels placeholder
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
    # Debug print (fixed)
    # ------------------
    print(
        f"sample_{i:06d}",
        voxels.shape,
        f"density={actual_density:.3f}",
        f"occupancy={voxel_density:.3f}",
        "saved"
    )