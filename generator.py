import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

BOX_SIZE = 10.0
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
# FIELD GENERATION (CLEAN PERIODIC VERSION)
# -----------------------------
def generate_field(params, res):

    x = np.linspace(0, 2*np.pi, res, endpoint=False)

    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    # keep periodic domain ONLY (important fix)
    field = compute_generalized_tpms(X, Y, Z, params)

    # light smoothing ONLY (avoid topology destruction)
    field = gaussian_filter(field, sigma=0.3)

    return field


# -----------------------------
# VOXELIZATION (NO MORPHOLOGY)
# -----------------------------
def field_to_voxels(field, threshold):

    # NO closing/opening → preserves monotonic density
    return field < threshold


# -----------------------------
# DENSITY (TRUE MONOTONIC METRIC)
# -----------------------------
def compute_density(voxels):
    return float(np.mean(voxels))


# -----------------------------
# THICKNESS SOLVER (ROBUST BISECTION)
# -----------------------------
def find_thickness(field, target_density, tol=0.01, max_iter=25):

    low, high = np.min(field), np.max(field)

    for _ in range(max_iter):

        mid = 0.5 * (low + high)

        voxels = field_to_voxels(field, mid)
        density = compute_density(voxels)

        if abs(density - target_density) < tol:
            return mid

        # monotonic assumption restored
        if density < target_density:
            low = mid
        else:
            high = mid

    return mid


# -----------------------------
# VOXEL → MESH (SAFE FOR ANSYS)
# -----------------------------
def voxels_to_mesh(voxels):

    pitch = BOX_SIZE / voxels.shape[0]

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
        voxels.astype(np.uint8),
        pitch=pitch
    )

    mesh.process(validate=True)

    return mesh


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def generate_lattice(params, target_density=0.2):

    field = generate_field(params, RESOLUTION)

    thickness = find_thickness(field, target_density)

    voxels = field_to_voxels(field, thickness)

    density = compute_density(voxels)

    mesh = voxels_to_mesh(voxels)

    return mesh, voxels, density, thickness