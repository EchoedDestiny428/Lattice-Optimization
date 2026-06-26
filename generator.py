import numpy as np
from scipy.ndimage import gaussian_filter

RESOLUTION = 20
BOX_SIZE = 10.0


# -----------------------------
# TPMS FIELDS
# -----------------------------
def gyroid(X, Y, Z):
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


def primitive(X, Y, Z):
    return np.cos(X) + np.cos(Y) + np.cos(Z)


def diamond(X, Y, Z):
    return (
        np.sin(X) * np.sin(Y) * np.sin(Z)
        + np.sin(X) * np.cos(Y) * np.cos(Z)
        + np.cos(X) * np.sin(Y) * np.cos(Z)
        + np.cos(X) * np.cos(Y) * np.sin(Z)
    )


def i_wp(X, Y, Z):
    return 2 * (
        np.cos(X) * np.cos(Y) + np.cos(Y) * np.cos(Z) + np.cos(Z) * np.cos(X)
    ) - (np.cos(2 * X) + np.cos(2 * Y) + np.cos(2 * Z))


def neovius(X, Y, Z):
    return 3 * (np.cos(X) + np.cos(Y) + np.cos(Z)) + 4 * np.cos(X) * np.cos(
        Y
    ) * np.cos(Z)


def generate_field(shape_type="gyroid", freq=1.0, noise=0.05):

    x = np.linspace(0, 2*np.pi*freq, RESOLUTION, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    if shape_type == "gyroid":
        field = gyroid(X, Y, Z)
    elif shape_type == "primitive":
        field = primitive(X, Y, Z)
    elif shape_type == "diamond":
        field = diamond(X, Y, Z)
    elif shape_type == "i_wp":
        field = i_wp(X, Y, Z)
    elif shape_type == "neovius":
        field = neovius(X, Y, Z)
    else:
        raise ValueError("unknown shape")

    field = (field - field.mean()) / (field.std() + 1e-8)

    field += noise * np.random.randn(*field.shape)

    return gaussian_filter(field, sigma=0.25)


# -----------------------------
# VOXELIZATION
# -----------------------------
def field_to_voxels(field, threshold):
    return field < threshold


def density(voxels):
    return float(voxels.mean())


# -----------------------------
# threshold solve
# -----------------------------
def find_threshold(field, target=0.3):
    lo, hi = -4.0, 4.0
    best_mid = 0

    for _ in range(30):
        mid = (lo + hi) / 2
        vox = field_to_voxels(field, mid)
        d = density(vox)
        best_mid = mid

        if abs(d - target) < 0.005:  # Tighter accuracy tolerance (0.5% max dev)
            return mid
        if d < target:
            lo = mid
        else:
            hi = mid
    return best_mid


# -----------------------------
# MAIN API
# -----------------------------
def generate_lattice(shape_type="gyroid", target_density=0.3, freq=1.0, noise=0.05):
    field = generate_field(shape_type, freq, noise)
    threshold = find_threshold(field, target_density)
    voxels = field_to_voxels(field, threshold)
    return voxels, density(voxels), threshold