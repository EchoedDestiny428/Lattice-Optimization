import os
import numpy as np
from mapdlgen import voxels_to_mapdl

DATASET_DIR = "data/samples"

for sample in os.listdir(DATASET_DIR):

    voxels = np.load(os.path.join(DATASET_DIR, sample, "voxels.npz"))["voxels"]

    mapdl = voxels_to_mapdl(voxels)

    mapdl.exit()