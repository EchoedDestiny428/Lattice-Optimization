import os
import numpy as np
from mapdlgen import voxels_to_mapdl, BOX_SIZE

DATASET_DIR = "data/samples"

for sample_name in sorted(os.listdir(DATASET_DIR)):

    sample_dir = os.path.join(DATASET_DIR, sample_name)
    voxel_path = os.path.join(sample_dir, "voxels.npy")

    if not os.path.isfile(voxel_path):
        continue

    print(f"\n[SIM] loading {voxel_path}")

    voxels = np.load(voxel_path)

    # -----------------------
    # MAPDL BUILD
    # -----------------------
    mapdl = voxels_to_mapdl(voxels)

    # -----------------------
    # BOUNDARY CONDITIONS
    # -----------------------

    # bottom fixed
    mapdl.nsel("S", "LOC", "Z", 0)
    mapdl.d("ALL", "ALL", 0)
    mapdl.allsel()

    # top compression
    mapdl.nsel("S", "LOC", "Z", BOX_SIZE)
    mapdl.d("ALL", "UZ", -0.1)
    mapdl.allsel()

    # -----------------------
    # SOLVE
    # -----------------------
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()

    # -----------------------
    # CLEAN EXIT
    # -----------------------
    mapdl.exit()

    print(f"[SIM] done {sample_name}")