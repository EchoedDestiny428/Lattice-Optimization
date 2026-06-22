import os
import numpy as np
from mapdlgen import voxels_to_mapdl

# ==========================
# LOAD VOXELS
# ==========================
sample_dir = "data/samples/sample_000000"

voxels = np.load(os.path.join(sample_dir, "voxels.npz"))["voxels"]

print("Voxel shape:", voxels.shape)

# ==========================
# BUILD MODEL
# ==========================
mapdl = voxels_to_mapdl(voxels)

print("Nodes:", mapdl.mesh.n_node)
print("Elements:", mapdl.mesh.n_elem)

nodes = mapdl.mesh.nodes
zmin, zmax = nodes[:, 2].min(), nodes[:, 2].max()

print("\nZ range:", zmin, zmax)

# geometry
BOX_SIZE = 10.0
L = BOX_SIZE
disp = 0.1

# ==========================
# SOLUTION
# ==========================
print("\nApplying BCs...")

mapdl.run("/SOLU")
mapdl.antype("STATIC")

# --------------------------
# bottom (fixed)
# --------------------------
mapdl.nsel("S", "LOC", "Z", zmin)
mapdl.d("ALL", "ALL", 0)
mapdl.cm("BOTTOM", "NODE")
mapdl.allsel()

# --------------------------
# top (displacement)
# --------------------------
mapdl.nsel("S", "LOC", "Z", zmax)
mapdl.d("ALL", "UZ", -disp)
mapdl.cm("TOP", "NODE")
mapdl.allsel()

# ==========================
# SOLVE
# ==========================
print("\nSolving...")
mapdl.solve()

# ==========================
# POSTPROCESSING
# ==========================
print("\nEntering POST1...")

mapdl.post1()
mapdl.set(1)

# ==========================
# REACTION FORCE (CORRECT)
# ==========================
mapdl.cmsel("S", "BOTTOM")

result = mapdl.run("FSUM")
text = str(result)

fz = None
for line in text.splitlines():
    if "FZ" in line:
        parts = line.replace("=", " ").split()
        for i, p in enumerate(parts):
            if p == "FZ":
                fz = float(parts[i + 1])
                break

if fz is None:
    raise RuntimeError("Could not extract FZ from FSUM")

print("\nTotal reaction FZ:", fz)

mapdl.allsel()

# ==========================
# TRUE HOMOGENIZATION (FIXED APPROACH)
# ==========================
strain = disp / L
stress = abs(fz) / (L * L)   # FIX: do NOT use voxel node area

E_eff = stress / strain

print("\n===== HOMOGENIZATION RESULT =====")
print("Reaction FZ:", fz)
print("Strain:", strain)
print("Stress (Pa):", stress)
print("Effective modulus (Pa):", E_eff)
print("Effective modulus (GPa):", E_eff / 1e9)

# ==========================
# CLEAN EXIT
# ==========================
input("\nPress Enter to close MAPDL...")
mapdl.finish()
mapdl.exit()