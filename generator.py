import numpy as np
from scipy.ndimage import gaussian_filter

RESOLUTION = 32
BOX_SIZE = 10.0


# -----------------------------
# TPMS FIELD (STABLE VERSION)
# -----------------------------
def tpms(X, Y, Z, p):
    c1, c2, c3, c4, c5 = p

    return (
        c1*np.sin(X)*np.cos(Y) +
        c2*np.sin(Y)*np.cos(Z) +
        c3*np.sin(Z)*np.cos(X) +
        c4*np.cos(X)*np.cos(Y)*np.cos(Z) +
        c5*(np.cos(2*X)+np.cos(2*Y)+np.cos(2*Z))
    )


# -----------------------------
# FIELD GENERATION (FIXED)
# -----------------------------
def generate_field(params):
    x = np.linspace(0, 2*np.pi, RESOLUTION, endpoint=False)

    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    field = tpms(X, Y, Z, params)

    # normalize → CRITICAL FIX
    field = (field - field.mean()) / (field.std() + 1e-8)

    # light smoothing only
    field = gaussian_filter(field, sigma=0.25)

    return field


# -----------------------------
# VOXELIZATION
# -----------------------------
def field_to_voxels(field, threshold):
    return field < threshold


# -----------------------------
# DENSITY
# -----------------------------
def density(voxels):
    return float(voxels.mean())


# -----------------------------
# THRESHOLD SOLVER (ROBUST)
# -----------------------------
def find_threshold(field, target=0.3):

    lo, hi = -2.0, 2.0   # FIXED DOMAIN (important!)

    for _ in range(25):
        mid = (lo + hi) / 2

        vox = field_to_voxels(field, mid)
        d = density(vox)

        if abs(d - target) < 0.01:
            return mid

        if d < target:
            lo = mid
        else:
            hi = mid

    return mid


# -----------------------------
# MAIN
# -----------------------------
def generate_lattice(params, target_density=0.3):

    field = generate_field(params)

    threshold = find_threshold(field, target_density)

    voxels = field_to_voxels(field, threshold)

    return voxels, density(voxels), threshold