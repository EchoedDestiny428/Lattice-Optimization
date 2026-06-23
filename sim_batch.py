import csv
import json
import os
import sys
import numpy as np
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

# --- LIVE CSV INITIALIZATION ---
CSV_HEADERS = [
    "sample_id",
    "shape",
    "shape_id",
    "target_density",
    "actual_density",
    "threshold",
    "freq",
    "noise",
    "resolution",
    "box_size_mm",
    "solid_voxels",
    "reaction_force_fz_n",
    "average_stress_pa",
    "applied_strain",
    "E_eff_pa",
    "E_eff_gpa",
    "status",
]

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(CSV_HEADERS)

print(f"Initialized live CSV ledger at: {OUTPUT_CSV}")

successful_solves = 0
failed_solves = 0

# ============================================================
# LAUNCH MAPDL ONCE (Persistent Instance)
# ============================================================
print("Launching single persistent MAPDL instance...")
global_mapdl = launch_mapdl(nproc=4)

# ============================================================
# BATCH PROCESSING LOOP
# ============================================================
try:
    for index, sample_id in enumerate(sample_ids):
        sample_path = os.path.join(DATASET_DIR, sample_id)

        voxel_file = os.path.join(sample_path, "voxels.npz")
        meta_file = os.path.join(sample_path, "metadata.json")

        if not (os.path.exists(voxel_file) and os.path.exists(meta_file)):
            print(
                f"[{index+1}/{len(sample_ids)}] Skipping {sample_id}: Missing files."
            )
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

        # Define common layout properties to pass to CSV regardless of pass/fail
        shape_id = metadata.get("shape_id", np.nan)  # Safe fallback if old json format

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
            mapdl.eqslv("PCG")
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

            print(
                f"-> Solved. Size: {box_size_mm}mm | E_eff: {E_eff / 1e9:.4f} GPa"
            )
            successful_solves += 1

            row_data = [
                sample_id,
                metadata["shape"],
                shape_id,
                metadata.get("target_density", 0.3),
                metadata["density"],
                metadata["threshold"],
                metadata["freq"],
                metadata["noise"],
                metadata["resolution"],
                box_size_mm,
                metadata["solid_voxels"],
                fz,
                stress,
                strain,
                E_eff,
                E_eff / 1e9,
                "SUCCESS",
            ]

        except Exception as e:
            print(f"!! Error solving {sample_id}: {str(e)}")
            failed_solves += 1

            row_data = [
                sample_id,
                metadata["shape"],
                shape_id,
                metadata.get("target_density", 0.3),
                metadata["density"],
                metadata["threshold"],
                metadata["freq"],
                metadata["noise"],
                metadata["resolution"],
                box_size_mm,
                metadata["solid_voxels"],
                np.nan,
                np.nan,
                strain,
                np.nan,
                np.nan,
                f"FAILED: {type(e).__name__}",
            ]

        # --- LIVE SAVE: Open file in append mode and flush straight to disk ---
        with open(OUTPUT_CSV, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

finally:
    print("\nClosing MAPDL instance...")
    try:
        global_mapdl.exit()
    except:
        pass

print("\n==============================================")
print(f"BATCH COMPLETE. Compilation saved to: {OUTPUT_CSV}")
print(f"Successful Solves: {successful_solves}")
print(f"Failed Solves:     {failed_solves}")
print("==============================================")