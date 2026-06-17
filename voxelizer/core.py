"""
core.py
-------
Orchestrates loading, grid calculations, geometric testing, and averaging.
"""

import numpy as np
from .loader import load_mesh
from .grid import compute_pitch, compute_grid_dims, build_sample_points
from .tester import test_points_inside


def stl_to_voxels(
    stl_path: str,
    resolution: int = 64,
    voxel_size: float | None = None,
    subsample: int = 4,
    batch_size: int = 500_000,
) -> np.ndarray:
    """
    High-level orchestration pipeline. Converts an STL file path directly
    into a continuous 3D density grid map of float32 values [0.0, 1.0].
    """
    # Load
    mesh = load_mesh(stl_path)

    # Compute Layout
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]

    pitch = compute_pitch(extents, resolution, voxel_size)
    grid_dims = compute_grid_dims(extents, pitch)
    points = build_sample_points(bounds, grid_dims, pitch, subsample)
    inside = test_points_inside(mesh, points, batch_size=batch_size)

    # Average Subgrids into continuous Density Values
    Nx, Ny, Nz = grid_dims
    S = subsample

    fill_grid = (
        inside
        .reshape(Nx, S, Ny, S, Nz, S)
        .mean(axis=(1, 3, 5))
        .astype(np.float32)
    )

    return fill_grid
