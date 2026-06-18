import os
import numpy as np

from mapdlgen import voxels_to_mapdl

sample_dir = "data/samples/sample_000000"

data = np.load(
    os.path.join(sample_dir, "voxels.npz")
)

voxels = data["voxels"]

print("Voxel shape:", voxels.shape)

mapdl = voxels_to_mapdl(voxels)

print("Nodes:", mapdl.mesh.n_node)
print("Elements:", mapdl.mesh.n_elem)

print("Applying BCs...")

mapdl.run("/SOLU")
mapdl.antype("STATIC")

# bottom fixed
mapdl.nsel("S", "LOC", "Z", 0)
mapdl.d("ALL", "ALL", 0)
mapdl.allsel()

# top displacement
mapdl.nsel("S", "LOC", "Z", 10)
mapdl.d("ALL", "UZ", -0.1)
mapdl.allsel()

print("Solving...")

output = mapdl.solve()

print(output)

print("SOLVE COMPLETE")

input("Press Enter to close MAPDL...")

mapdl.exit()