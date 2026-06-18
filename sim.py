import os
import numpy as np
from mapdlgen import voxels_to_mapdl

DATASET_DIR = "data/samples"

for name in sorted(os.listdir(DATASET_DIR)):

    sample_dir = os.path.join(DATASET_DIR, name)

    voxel_path = os.path.join(sample_dir, "voxels.npz")

    if not os.path.exists(voxel_path):
        continue

    print(f"\nLoading {name}")

    data = np.load(voxel_path)

    print("Keys:", data.files)

    voxels = data["voxels"]

    print("Voxel shape:", voxels.shape)

    mapdl = voxels_to_mapdl(voxels)

    print("MAPDL model created")