import numpy as np
from scipy.ndimage import gaussian_filter

BOX_SIZE = 10.0
RESOLUTION = 20


def tpms(X, Y, Z, p):
    c1, c2, c3, c4, c5 = p

    return (
        c1*np.sin(X)*np.cos(Y)
        + c2*np.sin(Y)*np.cos(Z)
        + c3*np.sin(Z)*np.cos(X)
        + c4*np.cos(X)*np.cos(Y)*np.cos(Z)
        + c5*(np.cos(2*X)+np.cos(2*Y)+np.cos(2*Z))
    )


def generate_field(params):

    x = np.linspace(
        0,
        2*np.pi,
        RESOLUTION,
        endpoint=False
    )

    X, Y, Z = np.meshgrid(
        x, x, x,
        indexing="ij"
    )

    field = tpms(X, Y, Z, params)

    field = (
        field - field.mean()
    ) / (
        field.std() + 1e-8
    )

    field = gaussian_filter(
        field,
        sigma=0.25
    )

    return field


def field_to_voxels(field, threshold):
    return field < threshold


def density(voxels):
    return float(voxels.mean())


def find_threshold(
    field,
    target=0.3,
    tol=0.005,
    max_iter=30
):

    lo = float(field.min())
    hi = float(field.max())

    best_mid = 0.0

    for _ in range(max_iter):

        mid = 0.5 * (lo + hi)

        vox = field_to_voxels(field, mid)

        d = density(vox)

        best_mid = mid

        if abs(d - target) < tol:
            return mid

        if d < target:
            lo = mid
        else:
            hi = mid

    return best_mid


def generate_lattice(
    params,
    target_density=0.3
):

    field = generate_field(params)

    threshold = find_threshold(
        field,
        target_density
    )

    voxels = field_to_voxels(
        field,
        threshold
    )

    return (
        voxels,
        density(voxels),
        threshold
    )