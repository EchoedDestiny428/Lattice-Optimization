import json
import os
import numpy as np
import trimesh
from generator import generate_density_matched_lattice, TARGET_DENSITY, BOX_SIZE, SOLID_VOLUME

VOXEL_RESOLUTION = 64
amount = 10

DATASET_DIR = os.path.join("data", "samples")
os.makedirs(DATASET_DIR, exist_ok=True)

for i in range(amount):

    sample_dir = os.path.join(
        DATASET_DIR,
        f"sample_{i:06d}"
    )
    os.makedirs(sample_dir, exist_ok=True)

    c1 = np.random.uniform(-1.5, 1.5)
    c2 = np.random.uniform(-1.5, 1.5)
    c3 = np.random.uniform(-1.5, 1.5)
    c4 = np.random.uniform(-1.0, 1.0)
    c5 = np.random.uniform(-0.5, 0.5)

    if abs(c1)+abs(c2)+abs(c3)+abs(c4)+abs(c5) < 0.2:
        c1 = 1.0

    params = [c1, c2, c3, c4, c5]

    mesh = generate_density_matched_lattice(
        params,
        target_density=TARGET_DENSITY
    )

    mesh.process(validate=True)
    volume = float(mesh.volume)
    surface_area = float(mesh.area)
    actual_density = volume / SOLID_VOLUME

    # ------------------
    # Voxelization
    # ------------------

    pitch = BOX_SIZE / VOXEL_RESOLUTION

    voxel_grid = mesh.voxelized(pitch)

    voxels = voxel_grid.matrix.astype(np.uint8)

    fixed_voxels = np.zeros(
        (VOXEL_RESOLUTION,
        VOXEL_RESOLUTION,
        VOXEL_RESOLUTION),
        dtype=np.uint8
    )

    sx = min(voxels.shape[0], VOXEL_RESOLUTION)
    sy = min(voxels.shape[1], VOXEL_RESOLUTION)
    sz = min(voxels.shape[2], VOXEL_RESOLUTION)

    fixed_voxels[:sx, :sy, :sz] = voxels[:sx, :sy, :sz]

    # ------------------
    # Save STL
    # ------------------

    stl_path = os.path.join(sample_dir, "lattice.stl")
    mesh.export(stl_path)

    np.savez_compressed(
        os.path.join(sample_dir, "voxels.npz"),
        voxels=fixed_voxels
    )

    # ------------------
    # Metadata
    # ------------------

    metadata = {
        "sample_id": f"sample_{i:06d}",

        "generation": {
            "target_density": TARGET_DENSITY
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
            "relative_density": actual_density
        },

        "voxelization": {
            "resolution": VOXEL_RESOLUTION
        },

        "box_size": BOX_SIZE
    }

    with open(
        os.path.join(sample_dir, "metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    # ------------------
    # Labels (empty for now)
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