import csv
import json
import os
import sys
import re
import numpy as np
from ansys.mapdl.core import launch_mapdl
from pathlib import Path

# Import constants and tools
from config import DATASET_DIR, SAMPLES_DIR, CSV_PATH, BOX_SIZE_MM
from src.mapdl_tools import voxels_to_mapdl

def simulate():
    # Ensure directory exists
    if not SAMPLES_DIR.exists():
        print(f"Error: Samples directory '{SAMPLES_DIR}' not found.")
        sys.exit(1)

    # Find all samples
    sample_ids = sorted([d.name for d in SAMPLES_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(sample_ids)} samples to process in {SAMPLES_DIR}")

    # CSV Headers
    CSV_HEADERS = [
        "sample_id", "topology", "complexity", "target_density", "actual_density", 
        "threshold", "freq", "noise", "resolution", "box_size_mm", "solid_voxels",
        "reaction_force_fz_n", "average_stress_pa", "applied_strain", 
        "E_eff_pa", "E_eff_gpa", "status"
    ]

    # Initialize CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

    print("Launching persistent MAPDL instance...")
    global_mapdl = launch_mapdl(nproc=4)

    try:
        for index, sample_id in enumerate(sample_ids):
            sample_path = SAMPLES_DIR / sample_id
            
            # --- UPDATE: Pointing to STL-derived voxels ---
            voxel_file = sample_path / "voxels_input.npz"
            meta_file = sample_path / "metadata.json"

            # Check if input file exists
            if not voxel_file.exists():
                print(f"Skipping {sample_id}: 'voxels_input.npz' not found. Run '03_voxelize_input.py' first.")
                continue
            
            if not meta_file.exists():
                print(f"Skipping {sample_id}: Metadata missing.")
                continue

            with open(meta_file, "r") as f:
                metadata = json.load(f)

            # Extraction logic
            complexity = metadata.get("complexity", 1)
            box_size = float(metadata.get("box_size", BOX_SIZE_MM))
            height = box_size / 1000.0  # Convert mm to m for SI consistency
            area = height * height
            strain = 0.01
            disp = strain * height
            
            # Loading the STl-derived input
            voxels = np.load(voxel_file)["voxels"]

            print(f"--- Processing [{index+1}/{len(sample_ids)}]: {sample_id} ---")

            try:
                # Map voxels to FEA
                mapdl = voxels_to_mapdl(global_mapdl, voxels, box_size=height)
                
                # Setup Simulation
                mapdl.run("/SOLU")
                mapdl.antype("STATIC")
                
                # Boundary Conditions Setup
                zmin = mapdl.mesh.nodes[:, 2].min()
                zmax = mapdl.mesh.nodes[:, 2].max()
                tol = (zmax - zmin) * 0.02
                
                # --- Step A: Setup & Lock Bottom Face ---
                mapdl.nsel("S", "LOC", "Z", zmin, zmin + tol)
                mapdl.cm("BottomNodes", "NODE")
                mapdl.d("ALL", "UZ", 0)
                
                # 3-2-1 Anchor to allow lateral expansion but prevent rigid-body rotation
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
                mapdl.solve()
                mapdl.post1()
                mapdl.set(1)
                
                # Extract Force cleanly
                mapdl.cmsel("S", "BottomNodes", "NODE")
                fsum = str(mapdl.run("FSUM"))
                fz = 0.0
                match = re.search(r"FZ\s*=\s*([-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?)", fsum, re.IGNORECASE)
                
                if match:
                    fz = float(match.group(1))
                else:
                    raise ValueError("Failed to extract FZ force.")

                stress = abs(fz) / area
                E_eff = stress / strain
                E_eff_gpa = E_eff / 1e9

                print(f"-> Success: {E_eff_gpa:.4f} GPa")
                status = "SUCCESS"
                
            except Exception as e:
                print(f"!! Error solving {sample_id}: {str(e)}")
                fz, stress, E_eff, E_eff_gpa = np.nan, np.nan, np.nan, np.nan
                status = f"FAILED: {type(e).__name__}"

            # Save to CSV
            row_data = [
                sample_id, metadata.get("topology", "Harmonic"), complexity, metadata.get("target_density", 0.3),
                metadata.get("actual_density", 0), metadata["threshold"], 
                metadata.get("freq", 0), metadata.get("noise", 0), 
                metadata["resolution"], box_size, metadata["solid_voxels"], 
                fz, stress, strain, E_eff, E_eff_gpa, status
            ]

            with open(CSV_PATH, mode="a", newline="") as f:
                csv.writer(f).writerow(row_data)

    finally:
        print("\nClosing MAPDL instance...")
        try: global_mapdl.exit()
        except: pass

    print(f"\nBATCH COMPLETE. Dataset compiled at: {CSV_PATH}")

if __name__ == "__main__":
    simulate()