import os
import numpy as np
from mapdlgen import voxels_to_mapdl

sample_dir = "data/samples/sample_000000"

data = np.load(os.path.join(sample_dir, "voxels.npz"))
voxels = data["voxels"]

print("Voxel shape:", voxels.shape)

mapdl = voxels_to_mapdl(voxels)

print("MAPDL model created")

print("Nodes:", mapdl.mesh.n_node)
print("Elements:", mapdl.mesh.n_elem)