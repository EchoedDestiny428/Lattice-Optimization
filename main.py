import os
import numpy as np
import trimesh
import json

OUTPUT_DIR = os.path.join("data", "generated_stl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_DENSITY = 0.20  
BOX_SIZE = 10.0
SOLID_VOLUME = BOX_SIZE ** 3

def compute_generalized_tpms(X, Y, Z, params):
    c1, c2, c3, c4, c5 = params
    term1 = c1 * np.sin(X) * np.cos(Y)
    term2 = c2 * np.sin(Y) * np.cos(Z)
    term3 = c3 * np.sin(Z) * np.cos(X)
    term4 = c4 * np.cos(X) * np.cos(Y) * np.cos(Z)
    term5 = c5 * (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z))
    return term1 + term2 + term3 + term4 + term5

def compute_density_for_thickness(params, thickness, res=40):
    x = np.linspace(0, 2 * np.pi, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    matrix = compute_generalized_tpms(X, Y, Z, params)
    binary_mask = np.abs(matrix) < thickness
    
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(binary_mask, pitch=BOX_SIZE/res)
    current_density = mesh.volume / SOLID_VOLUME
    return mesh, current_density

def generate_density_matched_lattice(params, target_density, tolerance=0.01):
    low = 0.01
    high = 3.0
    
    for _ in range(15):
        mid_thickness = (low + high) / 2
        mesh, current_density = compute_density_for_thickness(params, mid_thickness, res=32)
        
        if abs(current_density - target_density) < tolerance:
            return compute_density_for_thickness(params, mid_thickness, res=45)[0]
            
        if current_density < target_density:
            low = mid_thickness
        else:
            high = mid_thickness
            
    return compute_density_for_thickness(params, mid_thickness, res=45)[0]

amount = 10

DATASET_DIR = os.path.join("data", "samples")
os.makedirs(DATASET_DIR, exist_ok=True)

for i in range(amount):

    sample_dir = os.path.join(
        DATASET_DIR,
        f"sample_{i:06d}"
    )
    os.makedirs(sample_dir, exist_ok=True)

    c1 = np.random.uniform(-1.5, 1.5)
    c2 = np.random.uniform(-1.5, 1.5)
    c3 = np.random.uniform(-1.5, 1.5)
    c4 = np.random.uniform(-1.0, 1.0)
    c5 = np.random.uniform(-0.5, 0.5)

    if abs(c1)+abs(c2)+abs(c3)+abs(c4)+abs(c5) < 0.2:
        c1 = 1.0

    params = [c1, c2, c3, c4, c5]

    mesh = generate_density_matched_lattice(
        params,
        target_density=TARGET_DENSITY
    )

    mesh.process(validate=True)

    # ------------------
    # Save STL
    # ------------------

    stl_path = os.path.join(
        sample_dir,
        "lattice.stl"
    )

    mesh.export(stl_path)

    # ------------------
    # Metadata
    # ------------------

    metadata = {
        "parameters": {
            "c1": float(c1),
            "c2": float(c2),
            "c3": float(c3),
            "c4": float(c4),
            "c5": float(c5)
        },
        "target_density": TARGET_DENSITY,
        "box_size": BOX_SIZE
    }

    with open(
        os.path.join(sample_dir, "metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    # ------------------
    # Labels (empty for now)
    # ------------------

    labels = {
        "relative_density": None,
        "effective_modulus": None,
        "peak_force": None,
        "ea": None,
        "sea": None
    }

    with open(
        os.path.join(sample_dir, "labels.json"),
        "w"
    ) as f:
        json.dump(labels, f, indent=4)