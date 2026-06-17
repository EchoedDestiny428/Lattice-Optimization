"""
grid.py
-------
Computes voxel grid dimensions and generates the full set of
sub-sample points used for fractional fill testing.
"""

import numpy as np


def compute_pitch(
    extents: np.ndarray,
    resolution: int,
    voxel_size: float | None,
) -> float:
    """
    Return the voxel edge length (pitch).
    """
    if voxel_size is not None:
        pitch = float(voxel_size)
        print(f"Mode     fixed voxel size = {pitch:.6g} units")
    else:
        pitch = float(extents.max()) / resolution
        print(f"Mode     resolution={resolution}  →  voxel size={pitch:.6g} units")

    return pitch


def compute_grid_dims(extents: np.ndarray, pitch: float) -> np.ndarray:
    """
    Return the integer grid dimensions (Nx, Ny, Nz).
    """
    dims = np.ceil(extents / pitch).astype(int)
    print(f"Grid     {dims[0]} × {dims[1]} × {dims[2]}  ({dims.prod():,} voxels)")
    return dims


def build_sample_points(
    bounds: np.ndarray,
    grid_dims: np.ndarray,
    pitch: float,
    subsample: int,
) -> np.ndarray:
    """
    Build the flat array of all (x, y, z) sample points.
    Each voxel is subdivided into a (subsample³) sub-grid.
    """
    Nx, Ny, Nz = grid_dims
    S = subsample
    origin = bounds[0]

    total = Nx * Ny * Nz * S ** 3
    print(f"Samples  {S}³ = {S**3} per voxel  →  {total:,} total points")

    # Evenly-spaced offsets centred inside each sub-cell
    half_step = pitch / (2 * S)
    offsets   = np.linspace(half_step, pitch - half_step, S)   # (S,)

    # Per-axis coordinates: voxel origin + sub-cell offset
    xs = (origin[0] + np.arange(Nx) * pitch)[:, None] + offsets  # (Nx, S)
    ys = (origin[1] + np.arange(Ny) * pitch)[:, None] + offsets  # (Ny, S)
    zs = (origin[2] + np.arange(Nz) * pitch)[:, None] + offsets  # (Nz, S)

    xs = xs.ravel()  # (Nx*S,)
    ys = ys.ravel()  # (Ny*S,)
    zs = zs.ravel()  # (Nz*S,)

    # Full meshgrid → column array of (x, y, z) triples
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])