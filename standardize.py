import os
import trimesh
import numpy as np

INPUT_STL_PATH = "converted_lattice.stl"
OUTPUT_STL_PATH = "standardized_lattice.stl"

def standardize_stl(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Could not find the input file at '{input_path}'")
        return

    print(f"Loading raw mesh geometry: {input_path}")
    mesh = trimesh.load(input_path)
    
    # 1. Uniformly scale so the maximum dimension fits the 0.95 footprint
    max_side = mesh.extents.max()
    scale_factor = 0.95 / max_side * 0.985
    mesh.apply_scale(scale_factor)
    
    # 2. CRITICAL FIX: Calculate center based strictly on spatial bounding box corners
    bbox_center = mesh.bounds.mean(axis=0)
    
    # Translate the bounding box center perfectly to the domain center
    target_center = np.array([0.5, 0.5, 0.5])
    translation_vector = target_center - bbox_center
    mesh.apply_translation(translation_vector)

    print("\n--- SYNCHRONIZED BOUNDING METRICS ---")
    print(f"  - New Bounding Box Bounds: \n{mesh.bounds}")
    print(f"  - New Bounding Box Center: {mesh.bounds.mean(axis=0)}")
    print(f"  - New Physical Extents:     {mesh.extents}")

    mesh.export(output_path)
    print(f"\nSUCCESS! Synchronized file saved to: '{output_path}'")

if __name__ == "__main__":
    standardize_stl(INPUT_STL_PATH, OUTPUT_STL_PATH)