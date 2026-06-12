import os
import trimesh
import numpy as np

# --- CONFIGURATION ---
INPUT_STL_PATH = "converted_lattice.stl"      # The unstandardized CAD file
OUTPUT_STL_PATH = "standardized_lattice.stl"  # The final clean reference file

def standardize_stl(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Could not find the input file at '{input_path}'")
        return

    print(f"Loading raw mesh geometry: {input_path}")
    mesh = trimesh.load(input_path)
    
    print("\n--- INITIAL GEOMETRY METRICS ---")
    print(f"  - Original Bounding Box Bounds: \n{mesh.bounds}")
    print(f"  - Original Physical Extents:     {mesh.extents}")
    print(f"  - Original Center of Mass:       {mesh.center_mass}")

    
    max_side = mesh.extents.max()
    scale_factor = 0.95 / max_side
    mesh.apply_scale(scale_factor)
    
    
    target_center = np.array([0.5, 0.5, 0.5])
    translation_vector = target_center - mesh.center_mass
    mesh.apply_translation(translation_vector)

    print("\n--- STANDARDIZED GEOMETRY METRICS ---")
    print(f"  - New Bounding Box Bounds: \n{mesh.bounds}")
    print(f"  - New Physical Extents:     {mesh.extents}")
    print(f"  - New Center of Mass:       {mesh.center_mass}")

    mesh.export(output_path)
    
    print("-" * 60)
    print(f"SUCCESS! Standardized geometry file saved to: '{output_path}'")
    print("This file is now locked to a [1.0 x 1.0 x 1.0] domain and perfectly centered.")
    print("-" * 60)

if __name__ == "__main__":
    standardize_stl(INPUT_STL_PATH, OUTPUT_STL_PATH)