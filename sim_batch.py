import json
import os
import sys
import numpy as np
import pandas as pd
from ansys.mapdl.core import launch_mapdl

from mapdlgen import voxels_to_mapdl

# ============================================================
# SETTINGS
# ============================================================
DATASET_DIR = "data/samples"
OUTPUT_CSV = "data/dataset_compiled.csv"

if not os.path.exists(DATASET_DIR):
    print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
    sys.exit(1)

sample_ids = sorted(
    [
        d
        for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]
)
print(f"Found {len(sample_ids)} samples to process.")

compiled_records = []

# ============================================================
# LAUNCH MAPDL ONCE (Persistent Instance)
# ============================================================
print("Launching single persistent MAPDL instance...")
global_mapdl = launch_mapdl()

# ============================================================
# BATCH PROCESSING LOOP
# ============================================================
try:
    for index, sample_id in enumerate(sample_ids):
        sample_path = os.path.join(DATASET_DIR, sample_id)

        voxel_file = os.path.join(sample_path, "voxels.npz")
        meta_file = os.path.join(sample_path, "metadata.json")

        if not (os.path.exists(voxel_file) and os.path.exists(meta_file)):
            print(f"[{index+1}/{len(sample_ids)}] Skipping {sample_id}: Missing files.")
            continue

        print(f"\n--- Processing [{index+1}/{len(sample_ids)}]: {sample_id} ---")

        with open(meta_file, "r") as f:
            metadata = json.load(f)

        box_size_mm = float(metadata.get("box_size", 10.0))
        height = box_size_mm / 1000.0  
        area = height * height  
        strain = 0.01  
        disp = strain * height  

        voxels = np.load(voxel_file)["voxels"]

        try:
            # Re-use the existing global_mapdl instance
            mapdl = voxels_to_mapdl(global_mapdl, voxels, box_size=height)

            # Apply Boundary Conditions
            mapdl.run("/SOLU")
            mapdl.antype("STATIC")

            zmin = mapdl.mesh.nodes[:, 2].min()
            zmax = mapdl.mesh.nodes[:, 2].max()

            # Fix Bottom Face
            mapdl.nsel("S", "LOC", "Z", zmin)
            mapdl.d("ALL", "UZ", 0)
            mapdl.d("ALL", "UX", 0)
            mapdl.d("ALL", "UY", 0)
            mapdl.allsel()

            # Displace Top Face
            mapdl.nsel("S", "LOC", "Z", zmax)
            mapdl.d("ALL", "UZ", -disp)
            mapdl.allsel()

            # Solve
            mapdl.solve()

            # Postprocessing (Text Parser)
            mapdl.post1()
            mapdl.set(1)
            mapdl.nsel("S", "LOC", "Z", zmin)

            result = mapdl.run("FSUM")
            text = str(result)

            fz = 0.0
            for line in text.splitlines():
                if "FZ" in line and "=" in line:
                    fz = float(line.split("=")[1])

            stress = abs(fz) / area
            E_eff = stress / strain

            print(f"-> Solved. Size: {box_size_mm}mm | E_eff: {E_eff / 1e9:.4f} GPa")

            record = {
                "sample_id": sample_id,
                "shape": metadata["shape"],
                "target_density": metadata.get("target_density", 0.3),
                "actual_density": metadata["density"],
                "threshold": metadata["threshold"],
                "freq": metadata["freq"],
                "noise": metadata["noise"],
                "resolution": metadata["resolution"],
                "box_size_mm": box_size_mm,
                "solid_voxels": metadata["solid_voxels"],
                "reaction_force_fz_n": fz,
                "average_stress_pa": stress,
                "applied_strain": strain,
                "E_eff_pa": E_eff,
                "E_eff_gpa": E_eff / 1e9,
                "status": "SUCCESS",
            }
            compiled_records.append(record)

        except Exception as e:
            print(f"!! Error solving {sample_id}: {str(e)}")
            record = {
                "sample_id": sample_id,
                "shape": metadata["shape"],
                "target_density": metadata.get("target_density", 0.3),
                "actual_density": metadata["density"],
                "threshold": metadata["threshold"],
                "freq": metadata["freq"],
                "noise": metadata["noise"],
                "resolution": metadata["resolution"],
                "box_size_mm": box_size_mm,
                "solid_voxels": metadata["solid_voxels"],
                "reaction_force_fz_n": np.nan,
                "average_stress_pa": np.nan,
                "applied_strain": strain,
                "E_eff_pa": np.nan,
                "E_eff_gpa": np.nan,
                "status": f"FAILED: {type(e).__name__}",
            }
            compiled_records.append(record)

finally:
    # CRITICAL: Shut down ANSYS once the entire batch loop is finished
    print("\nClosing MAPDL instance...")
    try:
        global_mapdl.exit()
    except:
        pass

# ============================================================
# EXPORT COMPILED DATASET
# ============================================================
df = pd.DataFrame(compiled_records)
df.to_csv(OUTPUT_CSV, index=False)

print("\n==============================================")
print(f"BATCH COMPLETE. Compilation saved to: {OUTPUT_CSV}")
print(f"Successful Solves: {df[df['status']=='SUCCESS'].shape[0]}")
print(f"Failed Solves:     {df[df['status']!='SUCCESS'].shape[0]}")
print("==============================================")