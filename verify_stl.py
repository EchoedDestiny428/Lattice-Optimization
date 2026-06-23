import os
import time
import torch
import torch.nn as nn
import numpy as np
import trimesh
from ansys.mapdl.core import launch_mapdl
from model import LatticeCNN3D
from mapdlgen import voxels_to_mapdl

STL_PATH = "test.stl"
MODEL_WEIGHTS_PATH = "lattice_cnn.pth"
RESOLUTION = 20                         # Strictly back to 20 for your trained AI weights
UNIVERSAL_BOX_SIZE_MM = 10.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(STL_PATH):
    print(f"Error: STL file not found at '{STL_PATH}'")
    exit(1)

print(f"Loading custom mesh: {STL_PATH}")
mesh = trimesh.load(STL_PATH)

max_dimension = mesh.extents.max()
scale_factor = UNIVERSAL_BOX_SIZE_MM / max_dimension
mesh.apply_scale(scale_factor)

print(f"Voxelizing into a strict {RESOLUTION}^3 spatial grid...")
# Sub-sampling trick: voxelize slightly inward to avoid edge-aliasing corner disconnections
voxel_grid = mesh.voxelized(pitch=UNIVERSAL_BOX_SIZE_MM / RESOLUTION)
voxels_matrix = voxel_grid.matrix[:RESOLUTION, :RESOLUTION, :RESOLUTION]

if voxels_matrix.shape != (RESOLUTION, RESOLUTION, RESOLUTION):
    padded = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION), dtype=bool)
    nx, ny, nz = voxels_matrix.shape
    padded[:nx, :ny, :nz] = voxels_matrix
    voxels_matrix = padded

actual_density = voxels_matrix.sum() / (RESOLUTION ** 3)
print(f"-> Voxelization complete. Solid Voxel Count: {voxels_matrix.sum()} / {RESOLUTION**3}")

print("\n[AI Inference] Loading 3D CNN pipeline...")
model = LatticeCNN3D().to(device)

if os.path.exists(MODEL_WEIGHTS_PATH):
    # Load weights strictly into the native model structure (No layer hacking)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    voxel_tensor = torch.tensor(voxels_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        predicted_stiffness = model(voxel_tensor)
    cnn_time = time.time() - start_time
    cnn_gpa = predicted_stiffness.cpu().item()
else:
    print(f"Error: '{MODEL_WEIGHTS_PATH}' not found.")
    exit(1)

print("\n[FEA Validation] Launching persistent MAPDL engine...")
mapdl = launch_mapdl(nproc=4)
fea_gpa = np.nan
fea_time = 0.0

try:
    start_time = time.time()
    height = UNIVERSAL_BOX_SIZE_MM / 1000.0
    area = height * height
    strain = 0.01
    disp = strain * height
    
    mapdl = voxels_to_mapdl(mapdl, voxels_matrix.astype(np.uint8), box_size=height)
    
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    
    zmin, zmax = mapdl.mesh.nodes[:, 2].min(), mapdl.mesh.nodes[:, 2].max()
    
    mapdl.nsel("S", "LOC", "Z", zmin)
    mapdl.d("ALL", "UZ", 0)
    mapdl.d("ALL", "UX", 0)
    mapdl.d("ALL", "UY", 0)
    mapdl.allsel()
    
    mapdl.nsel("S", "LOC", "Z", zmax)
    mapdl.d("ALL", "UZ", -disp)
    mapdl.allsel()
    
    print("Solving finite element system via MAPDL sparse solver...")
    mapdl.solve()
    
    mapdl.post1()
    mapdl.set(1)
    mapdl.nsel("S", "LOC", "Z", zmin)
    result = mapdl.run("FSUM")
    
    fz = 0.0
    for line in str(result).splitlines():
        if "FZ" in line and "=" in line:
            fz = float(line.split("=")[1])
            
    stress = abs(fz) / area
    fea_gpa = (stress / strain) / 1e9
    fea_time = time.time() - start_time
    
except Exception as e:
    print(f"!! MAPDL solver error encountered: {e}")
finally:
    print("Safely exiting MAPDL kernel...")
    mapdl.exit()

print("\n" + "="*50)
print("         CROSS-VALIDATION RESULTS SUMMARY         ")
print("="*50)
print(f"Target File:          {os.path.basename(STL_PATH)}")
print(f"Discretization Grid:  {RESOLUTION}x{RESOLUTION}x{RESOLUTION}")
print(f"Solid Voxel Count:    {voxels_matrix.sum()} / {RESOLUTION**3}")
print("-"*50)
print(f"3D CNN Prediction:    {cnn_gpa:.4f} GPa  (Time: {cnn_time*1000:.2f} ms)")
if not np.isnan(fea_gpa):
    print(f"PyAnsys FEA Solve:    {fea_gpa:.4f} GPa  (Time: {fea_time:.2f} seconds)")
    absolute_error = abs(cnn_gpa - fea_gpa)
    percent_error = (absolute_error / fea_gpa) * 100
    print("-"*50)
    print(f"Absolute Deviation:   {absolute_error:.4f} GPa")
    print(f"AI Prediction Error:  {percent_error:.2f}%")
    print(f"AI Speedup Factor:    {fea_time / (cnn_time + 1e-8):.1f}x Faster")
else:
    print("PyAnsys FEA Solve:    FAILED")
print("="*50)