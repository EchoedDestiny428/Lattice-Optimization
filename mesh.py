from pathlib import Path
import numpy as np
import trimesh


def load_mesh(path: str | Path) -> "trimesh.Trimesh":
    """Load an STL (or any supported format) and return a Trimesh object."""
    mesh = trimesh.load(str(path), force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not read a valid triangle mesh from: {path}")

    if not mesh.is_watertight:
        print(
            "⚠  Mesh is NOT watertight (open edges detected).\n"
            "   Inside/outside tests may be unreliable near those regions."
        )

    return mesh


def mesh_info(mesh: "trimesh.Trimesh") -> tuple[np.ndarray, np.ndarray]:
    """
    Return (bounds, extents) for a mesh.

    bounds  : shape (2, 3) — [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    extents : shape (3,)   — [size_x, size_y, size_z]
    """
    bounds  = mesh.bounds
    extents = bounds[1] - bounds[0]
    return bounds, extents