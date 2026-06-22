import os
import sys
import json
import numpy as np

from mapdlgen import voxels_to_mapdl


# ==========================
# LOAD DATA
# ==========================
sample_dir = "data/samples/sample_000000"

with open(os.path.join(sample_dir, "metadata.json")) as f:
    meta = json.load(f)

voxels = np.load(os.path.join(sample_dir, "voxels.npz"))["voxels"]

print("Voxel shape:", voxels.shape)

L = float(meta["box_size"])
disp = 0.1


# ==========================
# BUILD MODEL
# ==========================
mapdl = voxels_to_mapdl(voxels, box_size=L)

print("Nodes:   ", mapdl.mesh.n_node)
print("Elements:", mapdl.mesh.n_elem)

nodes = mapdl.mesh.nodes
zmin = nodes[:, 2].min()
zmax = nodes[:, 2].max()

print("\nZ range:", zmin, "→", zmax)


# ============================================================
# SOLUTION
# ============================================================
print("\nApplying BCs...")

mapdl.prep7()
mapdl.finish()
mapdl.run("/SOLU")
mapdl.antype("STATIC")


# ============================================================
# BOTTOM BOUNDARY (only UZ fixed)
# ============================================================
tol = 1e-8 * L
bottom_mask = np.abs(nodes[:, 2] - zmin) < tol
bottom_nodes = nodes[bottom_mask]

bottom_ids = bottom_nodes[:, 0].astype(int)

mapdl.nsel("S", "LOC", "Z", zmin)
mapdl.d("ALL", "UZ", 0)


# ============================================================
# REMOVE RIGID BODY MODES (stable method)
# ============================================================
mapdl.nsel("S", "LOC", "Z", zmin)
mapdl.nsel("R", "LOC", "X", nodes[:, 0].min(), tol)
n_rb1 = mapdl.get_value("NODE", 0, "NUM", "MIN")

mapdl.nsel("S", "LOC", "Z", zmin)
mapdl.nsel("R", "LOC", "X", nodes[:, 0].max(), tol)
n_rb2 = mapdl.get_value("NODE", 0, "NUM", "MIN")

mapdl.d(n_rb1, "UX", 0)
mapdl.d(n_rb1, "UY", 0)
mapdl.d(n_rb2, "UY", 0)

mapdl.allsel()


# ============================================================
# TOP DISPLACEMENT
# ============================================================
top_mask = np.abs(nodes[:, 2] - zmax) < tol
top_nodes = nodes[top_mask]

mapdl.nsel("S", "LOC", "Z", zmax)
mapdl.d("ALL", "UZ", -disp)
mapdl.allsel()


# ============================================================
# SOLVE
# ============================================================
print("\nSolving...")
mapdl.solve()


# ============================================================
# POSTPROCESSING
# ============================================================
print("\nEntering POST1...")

mapdl.post1()
mapdl.set(1)


# ============================================================
# REACTION FORCE
# ============================================================
mapdl.nsel("S", "LOC", "Z", zmin)

# Sum the nodal forces for the currently selected node set
mapdl.fsum()
# Safely extract the cumulative Z reaction force from the FSUM command buffer
fz = mapdl.get_value("FSUM", 0, "ITEM", "Z")

mapdl.allsel()

print("\nTotal reaction FZ:", fz)


# ============================================================
# HOMOGENIZATION (CORRECT FORM)
# ============================================================
strain = disp / L

# use full cross-sectional area (RVE definition)
stress = abs(fz) / (L * L)

E_eff = stress / strain


# ============================================================
# OUTPUT
# ============================================================
print("\n===== HOMOGENIZATION RESULT =====")
print("Reaction FZ        :", fz)
print("Strain             :", strain)
print("Stress (Pa)        :", stress)
print("E_eff (Pa)         :", E_eff)
print("E_eff (GPa)        :", E_eff / 1e9)


# ============================================================
# CLEAN EXIT
# ============================================================
if sys.stdin.isatty():
    input("\nPress Enter to close MAPDL...")

mapdl.finish()
mapdl.exit()