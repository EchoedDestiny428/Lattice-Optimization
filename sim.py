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

print("Z range:")
print(mapdl.mesh.nodes[:,2].min())
print(mapdl.mesh.nodes[:,2].max())

print("Applying BCs...")

mapdl.run("/SOLU")
mapdl.antype("STATIC")

# bottom fixed
mapdl.nsel("S", "LOC", "Z", 0)
print("bottom nodes:", mapdl.mesh.n_node)

mapdl.d("ALL", "ALL", 0)
mapdl.allsel()

# top displacement
mapdl.nsel("S", "LOC", "Z", 10)
print("top nodes:", mapdl.mesh.n_node)

mapdl.d("ALL", "UZ", -0.1)
mapdl.allsel()

print("Solving...")

output = mapdl.solve()

print(output)

# ==========================
# DEBUG RESULTS
# ==========================

print("Entering POST1...")

mapdl.post1()
mapdl.set(1)

# bottom surface
mapdl.nsel("S", "LOC", "Z", 0)

# safest + simplest
result = mapdl.run("FSUM")
text = str(result)
print(text)

for line in text.splitlines():
    if "FZ" in line:
        fz = float(line.split("=")[1])
        print("FZ:", fz)

height = 10.0
area = 10.0 * 10.0
disp = 0.1

strain = disp / height
stress = abs(fz) / area

E_eff = stress / strain

print("Effective modulus:", E_eff)

mapdl.allsel()

print("SOLVE COMPLETE")
input("Press Enter to close MAPDL...")
mapdl.exit()