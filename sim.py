import os
import sys
import numpy as np

from mapdlgen import voxels_to_mapdl

# ============================================================
# ANALYSIS PARAMETERS (SI Units: Meters, Newtons, Pascals)
# ============================================================
height = 0.01  # 10 mm in meters
area = height * height  # Cross-sectional area (m^2)
disp = 0.0001  # 0.1 mm compression in meters

sample_dir = "data/samples/sample_000000"

data = np.load(os.path.join(sample_dir, "voxels.npz"))
voxels = data["voxels"]

print("Voxel shape:", voxels.shape)

# Synchronize geometry by passing explicit box physical size
mapdl = voxels_to_mapdl(voxels, box_size=height)

print("Nodes:   ", mapdl.mesh.n_node)
print("Elements:", mapdl.mesh.n_elem)

zmin = mapdl.mesh.nodes[:, 2].min()
zmax = mapdl.mesh.nodes[:, 2].max()
print(f"Z range: {zmin} → {zmax}")

# ============================================================
# SOLUTIONS & BOUNDARY CONDITIONS
# ============================================================
print("Applying BCs...")
mapdl.run("/SOLU")
mapdl.antype("STATIC")

# 1. Bottom Face Constraints (Fully lock face to prevent sliding/spinning)
mapdl.nsel("S", "LOC", "Z", zmin)
mapdl.d("ALL", "UZ", 0)
mapdl.d("ALL", "UX", 0)
mapdl.d("ALL", "UY", 0)
mapdl.allsel()

# 2. Top Face Compression Displacement
mapdl.nsel("S", "LOC", "Z", zmax)
mapdl.d("ALL", "UZ", -disp)
mapdl.allsel()

print("Solving...")
output = mapdl.solve()
print(output)

# ============================================================
# POSTPROCESSING & HOMOGENIZATION (REVERTED TO WORKING TEXT PARSER)
# ============================================================
print("Entering POST1...")
mapdl.post1()
mapdl.set(1)

# Select bottom surface nodes
mapdl.nsel("S", "LOC", "Z", zmin)

# Execute FSUM and capture the raw string output directly
result = mapdl.run("FSUM")
text = str(result)
print(text)

fz = 0.0
# Parse the text block line by line for the exact FZ numeric float
for line in text.splitlines():
    if "FZ" in line and "=" in line:
        # Splits '  FZ  =  -1.2345E+05' at '=' and converts right side to float
        fz = float(line.split("=")[1])
        print("Extracted FZ:", fz)

mapdl.allsel()

print("\n===== HOMOGENIZATION RESULTS =====")
print(f"Reaction Force FZ (N): {fz:.4e}")

strain = disp / height
stress = abs(fz) / area
E_eff = stress / strain

print(f"Applied Strain       : {strain:.6f}")
print(f"Average Stress (Pa)  : {stress:.4e}")
print(f"E_eff (Pa)           : {E_eff:.4e}")
print(f"E_eff (GPa)          : {E_eff / 1e9:.4f}")
print("==================================\n")

print("SOLVE COMPLETE")
if sys.stdin.isatty():
    input("Press Enter to close MAPDL...")

mapdl.exit()