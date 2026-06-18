import numpy as np
from scipy.ndimage import gaussian_filter

# -----------------------------
# GLOBAL SETTINGS
# -----------------------------
RESOLUTION = 20
BOX_SIZE = 10.0


# -----------------------------
# TPMS BASE FUNCTIONS
# -----------------------------
def gyroid(X, Y, Z):
    return (
        np.sin(X)*np.cos(Y) +
        np.sin(Y)*np.cos(Z) +
        np.sin(Z)*np.cos(X)
    )


def diamond(X, Y, Z):
    return (
        np.sin(X)*np.sin(Y)*np.sin(Z) +
        np.sin(X)*np.cos(Y)*np.cos(Z) +
        np.cos(X)*np.sin(Y)*np.cos(Z) +
        np.cos(X)*np.cos(Y)*np.sin(Z)
    )


def primitive(X, Y, Z):
    return (
        np.cos(X) +
        np.cos(Y) +
        np.cos(Z)
    )


# -----------------------------
# FIELD GENERATION
# -----------------------------
def generate_field(shape_type="gyroid", freq=1.0, noise=0.05):

    x = np.linspace(0, 2*np.pi*freq, RESOLUTION, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    if shape_type == "gyroid":
        field = gyroid(X, Y, Z)
    elif shape_type == "diamond":
        field = diamond(X, Y, Z)
    elif shape_type == "primitive":
        field = primitive(X, Y, Z)
    else:
        raise ValueError(f"Unknown shape: {shape_type}")

    # normalize for stable thresholding
    field = (field - field.mean()) / (field.std() + 1e-8)

    # small randomness (optional)
    field += noise * np.random.randn(*field.shape)

    field = gaussian_filter(field, sigma=0.25)

    return field


# -----------------------------
# VOXELIZATION
# -----------------------------
def field_to_voxels(field, threshold):
    return field < threshold


def density(voxels):
    return float(voxels.mean())


# -----------------------------
# THRESHOLD SEARCH
# -----------------------------
def find_threshold(field, target_density=0.3, tol=0.01, max_iter=25):

    lo, hi = -2.0, 2.0
    best_mid = 0.0

    for _ in range(max_iter):

        mid = 0.5 * (lo + hi)
        vox = field_to_voxels(field, mid)
        d = density(vox)

        best_mid = mid

        if abs(d - target_density) < tol:
            return mid

        if d < target_density:
            lo = mid
        else:
            hi = mid

    return best_mid


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def generate_lattice(shape_type="gyroid",
                      target_density=0.3,
                      freq=1.0,
                      noise=0.05):

    field = generate_field(shape_type, freq, noise)

    threshold = find_threshold(field, target_density)

    voxels = field_to_voxels(field, threshold)

    return voxels, density(voxels), threshold