import os
import sys
import re
import time
import numpy as np
import torch
from ansys.mapdl.core import launch_mapdl

# Import your centralized config and tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RESOLUTION, BOX_SIZE_MM, MODEL_WEIGHTS_PATH, DEVICE
from src.model import LatticeCNN3D
from src.mapdl_tools import voxels_to_mapdl
from src.voxelizer import voxelize_stl

# Configuration
STL_PATH = "test2.stl"

if not os.path.exists(STL_PATH):
    raise FileNotFoundError(f"STL file not found: {STL_PATH}")

print(f"--- Processing: {STL_PATH} ---")
print(f"Resolution target: {RESOLUTION} | Box size: {BOX_SIZE_MM}mm")

# ============================================================
# 1. AI Inference
# ============================================================
print("\n[AI Inference] Running CNN prediction...")
voxels_matrix, actual_density = voxelize_stl(
    STL_PATH, resolution=RESOLUTION, box_size_mm=BOX_SIZE_MM,
)

model = LatticeCNN3D(resolution=RESOLUTION).to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# Prep tensor
voxel_tensor = torch.tensor(voxels_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

start_time = time.time()
with torch.no_grad():
    log_prediction = model(voxel_tensor).item()
    cnn_gpa = 10 ** log_prediction

cnn_time = time.time() - start_time


# ============================================================
# 2. FEA Validation
# ============================================================
print("[FEA Validation] Launching MAPDL instance...")
mapdl = launch_mapdl(nproc=4)
fea_gpa = np.nan
fea_time = 0.0

try:
    start_time = time.time()
    height = BOX_SIZE_MM / 1000.0  # Convert to Meters
    area = height * height
    strain = 0.01
    disp = strain * height

    # Map voxels to FEA
    mapdl = voxels_to_mapdl(mapdl, voxels_matrix.astype(np.uint8), box_size=height)
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")

    # BCs Definition
    zmin = mapdl.mesh.nodes[:, 2].min()
    zmax = mapdl.mesh.nodes[:, 2].max()
    tol = (zmax - zmin) * 0.02

    # --- Step A: Lock Bottom Face ---
    mapdl.nsel("S", "LOC", "Z", zmin, zmin + tol)
    mapdl.cm("BottomNodes", "NODE")
    mapdl.d("ALL", "UZ", 0)
    
    # Anchor to allow lateral expansion but prevent rigid-body rotation
    nodes_at_bottom = mapdl.mesh.nodes
    bottom_indices = np.where((nodes_at_bottom[:, 2] >= zmin) & (nodes_at_bottom[:, 2] <= zmin + tol))[0]
    
    if len(bottom_indices) > 0:
        x_mid = (nodes_at_bottom[:, 0].max() + nodes_at_bottom[:, 0].min()) / 2
        y_mid = (nodes_at_bottom[:, 1].max() + nodes_at_bottom[:, 1].min()) / 2
        distances = (nodes_at_bottom[bottom_indices, 0] - x_mid)**2 + (nodes_at_bottom[bottom_indices, 1] - y_mid)**2
        center_node_id = mapdl.mesh.enum[bottom_indices[np.argmin(distances)]]
        
        mapdl.d(int(center_node_id), "UX", 0)
        mapdl.d(int(center_node_id), "UY", 0)
    
    mapdl.allsel()

    # --- Step B: Displace Top Face ---
    mapdl.nsel("S", "LOC", "Z", zmax - tol, zmax)
    mapdl.d("ALL", "UZ", -disp)
    mapdl.allsel()

    # --- Step C: Solve ---
    print("Solving FE system via MAPDL sparse solver...")
    mapdl.solve()
    
    # --- Step D: Post-Processing & Force Summation ---
    mapdl.post1()
    mapdl.set(1)
    
    # Select the pre-saved component group cleanly
    mapdl.cmsel("S", "BottomNodes", "NODE")

    # Force Summation
    fsum = str(mapdl.run("FSUM"))
    match = re.search(r"FZ\s*=\s*([-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?)", fsum, re.IGNORECASE)

    if match:
        fz = float(match.group(1))
        fea_gpa = ((abs(fz) / area) / strain) / 1e9
        fea_time = time.time() - start_time
    else:
        print("!! Failed to extract FZ force.")

except Exception as e:
    print(f"!! MAPDL solver error: {e}")
finally:
    mapdl.exit()


# ============================================================
# 3. Report
# ============================================================
print("\n" + "=" * 50)
print(f"         CROSS-VALIDATION RESULTS")
print("=" * 50)
print(f"3D CNN Prediction:    {cnn_gpa:.4f} GPa (Time: {cnn_time*1000:.2f} ms)")

if not np.isnan(fea_gpa):
    print(f"PyAnsys FEA Solve:    {fea_gpa:.4f} GPa (Time: {fea_time:.2f} seconds)")
    error = abs(cnn_gpa - fea_gpa) / fea_gpa * 100.0
    print(f"AI Prediction Error:  {error:.2f}%")
else:
    print("FEA Solve failed.")
print("=" * 50)