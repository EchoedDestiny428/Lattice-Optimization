import os
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

OUTPUT_DIR = os.path.join("data", "generated_stl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOX_SIZE = 10.0
SOLID_VOLUME = BOX_SIZE ** 3
RESOLUTION = 64


# -----------------------------
# TPMS FIELD
# -----------------------------
def compute_generalized_tpms(X, Y, Z, params):
    c1, c2, c3, c4, c5 = params

    return (
        c1 * np.sin(X) * np.cos(Y) +
        c2 * np.sin(Y) * np.cos(Z) +
        c3 * np.sin(Z) * np.cos(X) +
        c4 * np.cos(X) * np.cos(Y) * np.cos(Z) +
        c5 * (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z))
    )


# -----------------------------
# FIELD GENERATION (CORE)
# -----------------------------
def generate_field(params, res):
    x = np.linspace(0, 2*np.pi, res, endpoint=False)

    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    # normalize to [0, BOX_SIZE]
    scale = BOX_SIZE / (2*np.pi)
    X *= scale
    Y *= scale
    Z *= scale

    field = compute_generalized_tpms(X, Y, Z, params)

    # smooth ONLY field (not binary)
    field = gaussian_filter(field, sigma=0.8)

    return field


# -----------------------------
# GEOMETRY FROM FIELD
# -----------------------------
def field_to_mesh(field, pitch):
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
        field,
        pitch=pitch
    )

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()

    return mesh


# -----------------------------
# DENSITY (STABLE METRIC)
# -----------------------------
def compute_density(field, threshold):
    binary = field < threshold

    # remove 1-voxel noise
    binary = binary_closing(binary, np.ones((2,2,2)))
    binary = binary_opening(binary, np.ones((2,2,2)))

    return binary.mean(), binary


# -----------------------------
# THICKNESS SOLVER (MONOTONIC)
# -----------------------------
def find_thickness(field, target_density, res, tol=0.01, max_iter=20):

    low, high = np.min(field), np.max(field)

    best_t = None
    best_diff = 1e9

    for _ in range(max_iter):

        t = 0.5 * (low + high)

        density, _ = compute_density(field, t)
        diff = density - target_density

        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_t = t

        if abs(diff) < tol:
            return t

        if density < target_density:
            low = t
        else:
            high = t

    return best_t


# -----------------------------
# MAIN GENERATION PIPELINE
# -----------------------------
def generate_lattice(params, target_density=0.2):

    pitch = BOX_SIZE / RESOLUTION
    field = generate_field(params, RESOLUTION)
    thickness = find_thickness(field, target_density, RESOLUTION)
    density, binary = compute_density(field, thickness)
    mesh = field_to_mesh(field, pitch)

    return mesh, binary, density, thickness