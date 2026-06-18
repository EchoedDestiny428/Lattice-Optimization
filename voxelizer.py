import numpy as np


def mesh_to_voxels(mesh, resolution=64):
    mesh = mesh.copy()

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation(-center)

    scale = 1.0 / np.max(mesh.extents)
    mesh.apply_scale(scale)

    coords = np.linspace(
        -0.5,
        0.5,
        resolution,
        endpoint=False
    )

    X, Y, Z = np.meshgrid(
        coords,
        coords,
        coords,
        indexing="ij"
    )

    points = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    ])

    # Occupancy query
    inside = mesh.contains(points)

    voxels = inside.reshape(
        resolution,
        resolution,
        resolution
    )

    return voxels.astype(np.uint8)