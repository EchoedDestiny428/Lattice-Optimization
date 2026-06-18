import os
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

OUTPUT_DIR = os.path.join("data", "generated_stl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_DENSITY = 0.20  
BOX_SIZE = 10.0
SOLID_VOLUME = BOX_SIZE ** 3
RESOLUTION = 32

def compute_generalized_tpms(X, Y, Z, params):
    c1, c2, c3, c4, c5 = params
    term1 = c1 * np.sin(X) * np.cos(Y)
    term2 = c2 * np.sin(Y) * np.cos(Z)
    term3 = c3 * np.sin(Z) * np.cos(X)
    term4 = c4 * np.cos(X) * np.cos(Y) * np.cos(Z)
    term5 = c5 * (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z))
    return term1 + term2 + term3 + term4 + term5

def compute_density_for_thickness(params, thickness, res=40):

    x = np.linspace(0, 2*np.pi, res, endpoint=False)

    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    X = X / (2*np.pi) * BOX_SIZE
    Y = Y / (2*np.pi) * BOX_SIZE
    Z = Z / (2*np.pi) * BOX_SIZE

    pitch = BOX_SIZE / res

    matrix = compute_generalized_tpms(X, Y, Z, params)

    matrix_smooth = gaussian_filter(matrix, sigma=0.8)

    binary_mask = matrix_smooth < thickness

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
        matrix_smooth,
        pitch=pitch
    )

    current_density = binary_mask.mean()

    return mesh, current_density, binary_mask

def generate_density_matched_lattice(params, target_density, tolerance=0.01):
    low = 0.01
    high = 3.0

    for _ in range(15):

        mid_thickness = (low + high) / 2

        mesh, current_density, binary_mask = (
            compute_density_for_thickness(
                params,
                mid_thickness,
                res=RESOLUTION
            )
        )

        if abs(current_density - target_density) < tolerance:

            mesh, current_density, binary_mask = (
                compute_density_for_thickness(
                    params,
                    mid_thickness,
                    res=RESOLUTION
                )
            )

            return (
                mesh,
                binary_mask,
                current_density,
                mid_thickness
            )

        if current_density < target_density:
            low = mid_thickness
        else:
            high = mid_thickness

    mesh, current_density, binary_mask = (
        compute_density_for_thickness(
            params,
            mid_thickness,
            res=RESOLUTION
        )
    )

    return (
        mesh,
        binary_mask,
        current_density,
        mid_thickness
    )