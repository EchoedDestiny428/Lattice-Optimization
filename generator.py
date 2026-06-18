import numpy as np
from scipy.ndimage import gaussian_filter

BOX_SIZE = 10.0
RESOLUTION = 32


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
# FIELD GENERATION
# -----------------------------
def generate_field(params, res):
    x = np.linspace(0, 2*np.pi, res, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    field = compute_generalized_tpms(X, Y, Z, params)

    # very light smoothing (do not destroy topology)
    field = gaussian_filter(field, sigma=0.3)

    return field


# -----------------------------
# VOXELIZATION
# -----------------------------
def field_to_voxels(field, threshold):
    return field < threshold


# -----------------------------
# DENSITY (MONOTONIC)
# -----------------------------
def compute_density(voxels):
    return float(voxels.mean())


# -----------------------------
# THICKNESS SOLVER
# -----------------------------
def find_thickness(field, target_density, tol=0.01, max_iter=25):

    low, high = field.min(), field.max()

    best_mid = None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)

        voxels = field_to_voxels(field, mid)
        density = compute_density(voxels)

        best_mid = mid

        if abs(density - target_density) < tol:
            return mid

        if density < target_density:
            low = mid
        else:
            high = mid

    return best_mid


# -----------------------------
# VOXEL → FEM GRID (IMPORTANT CHANGE)
# -----------------------------
def voxels_to_fem_grid(voxels):
    """
    Converts voxel occupancy directly into FEM-ready data.
    Each voxel = potential SOLID185 element.
    """

    voxels = voxels.astype(np.uint8)

    # element size
    pitch = BOX_SIZE / voxels.shape[0]

    return {
        "voxels": voxels,
        "pitch": pitch,
        "shape": voxels.shape
    }


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def generate_lattice(params, target_density=0.2):

    field = generate_field(params, RESOLUTION)

    thickness = find_thickness(field, target_density)

    voxels = field_to_voxels(field, thickness)

    density = compute_density(voxels)

    fem_grid = voxels_to_fem_grid(voxels)

    return fem_grid, voxels, density, thickness