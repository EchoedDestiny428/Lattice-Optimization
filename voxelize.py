"""
voxelize.py — Core voxelization logic.

Pipeline
--------
1. compute_pitch     — decide voxel edge length from resolution or physical size
2. build_sample_pts  — generate sub-sample test points inside every voxel
3. test_inside       — ask the mesh which points are interior (batched)
4. aggregate_fill    — average sub-samples per voxel → 0-1 fill fraction
5. stl_to_voxels     — convenience wrapper that runs the full pipeline
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import trimesh
from .mesh import load_mesh, mesh_info


# ---------------------------------------------------------------------------
# Step 1 — pitch
# ---------------------------------------------------------------------------

def compute_pitch(
    extents: np.ndarray,
    resolution: int,
    voxel_size: float | None,
) -> float:
    """
    Return the voxel edge length (pitch).

    If voxel_size is given it is used directly.
    Otherwise pitch = longest_extent / resolution.
    """
    if voxel_size is not None:
        return float(voxel_size)
    return float(extents.max()) / resolution


# ---------------------------------------------------------------------------
# Step 2 — sample-point grid
# ---------------------------------------------------------------------------

def build_sample_pts(
    bounds: np.ndarray,
    grid_dims: np.ndarray,
    pitch: float,
    subsample: int,
) -> np.ndarray:
    """
    Build the full array of test points, shape (Nx*S * Ny*S * Nz*S, 3).

    Each voxel is divided into a subsample³ sub-grid.  Points are centred
    within their sub-cell so the coverage is evenly spread.
    """
    Nx, Ny, Nz = grid_dims
    S = subsample
    origin = bounds[0]

    half_step = pitch / (2 * S)
    offsets   = np.linspace(half_step, pitch - half_step, S)   # S values in [0, pitch)

    # All x/y/z coords: voxel origins + sub-sample offsets
    xs = (origin[0] + np.arange(Nx) * pitch)[:, None] + offsets[None, :]   # (Nx, S)
    ys = (origin[1] + np.arange(Ny) * pitch)[:, None] + offsets[None, :]   # (Ny, S)
    zs = (origin[2] + np.arange(Nz) * pitch)[:, None] + offsets[None, :]   # (Nz, S)

    xx, yy, zz = np.meshgrid(xs.ravel(), ys.ravel(), zs.ravel(), indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


# ---------------------------------------------------------------------------
# Step 3 — inside/outside test
# ---------------------------------------------------------------------------

def test_inside(
    mesh: "trimesh.Trimesh",
    points: np.ndarray,
    batch_size: int,
    verbose: bool,
) -> np.ndarray:
    """
    Test which points are inside the mesh, processing in batches.

    Returns a bool array of length len(points).
    """
    total  = len(points)
    inside = np.empty(total, dtype=bool)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        inside[start:end] = mesh.contains(points[start:end])

        if verbose:
            print(f"  {end / total * 100:5.1f}%", end="\r", flush=True)

    if verbose:
        print("  100.0%  done            ")

    return inside


# ---------------------------------------------------------------------------
# Step 4 — aggregate sub-samples into per-voxel fill fractions
# ---------------------------------------------------------------------------

def aggregate_fill(
    inside: np.ndarray,
    grid_dims: np.ndarray,
    subsample: int,
) -> np.ndarray:
    """
    Reshape the flat inside-test results and average over sub-samples.

    Returns float32 array of shape (Nx, Ny, Nz) with values in [0, 1].
    """
    Nx, Ny, Nz = grid_dims
    S = subsample

    return (
        inside
        .reshape(Nx, S, Ny, S, Nz, S)
        .mean(axis=(1, 3, 5))
        .astype(np.float32)
    )


# ---------------------------------------------------------------------------
# Step 5 — public API
# ---------------------------------------------------------------------------

def stl_to_voxels(
    stl_path: str | Path,
    resolution: int = 64,
    voxel_size: float | None = None,
    subsample: int = 4,
    batch_size: int = 500_000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Convert an STL file to a 3-D voxel grid of fill values.

    Parameters
    ----------
    stl_path   : Path to the .stl file.
    resolution : Voxels along the longest axis (ignored when voxel_size set).
    voxel_size : Physical voxel edge length in mesh units (overrides resolution).
    subsample  : Sub-samples per axis per voxel — higher = sharper surfaces.
                 4 → 64 samples/voxel (default).  8 → 512 samples/voxel.
    batch_size : Points per inside-test batch (tune if you hit memory limits).
    verbose    : Print progress to stdout.

    Returns
    -------
    np.ndarray — shape (Nx, Ny, Nz), dtype float32, values in [0.0, 1.0].
    """
    mesh          = load_mesh(stl_path)
    bounds, extents = mesh_info(mesh)
    pitch         = compute_pitch(extents, resolution, voxel_size)
    grid_dims     = np.ceil(extents / pitch).astype(int)

    if verbose:
        Nx, Ny, Nz = grid_dims
        S = subsample
        mode = (
            f"voxel_size={pitch:.6g} units"
            if voxel_size is not None
            else f"resolution={resolution}  →  voxel_size={pitch:.6g} units"
        )
        print(f"Mode     {mode}")
        print(f"Grid     {Nx} × {Ny} × {Nz}  ({Nx*Ny*Nz:,} voxels)")
        print(f"Samples  {S}³ = {S**3} per voxel  →  {Nx*Ny*Nz * S**3:,} total")
        print("Testing inside/outside …")

    sample_pts = build_sample_pts(bounds, grid_dims, pitch, subsample)
    inside     = test_inside(mesh, sample_pts, batch_size, verbose)
    fill       = aggregate_fill(inside, grid_dims, subsample)

    return fill