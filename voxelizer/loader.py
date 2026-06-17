"""
loader.py
---------
Loads and validates an STL file into a trimesh.Trimesh object.
"""

import trimesh
import numpy as np


def load_mesh(stl_path: str) -> trimesh.Trimesh:
    """
    Load an STL file and return a validated Trimesh.
    """
    print(f"Loading  {stl_path} …")
    mesh = trimesh.load(str(stl_path), force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Could not read a valid triangle mesh from: {stl_path}"
        )

    bounds  = mesh.bounds           # shape (2, 3)
    extents = bounds[1] - bounds[0]
    print(f"Bounds   min={bounds[0]}  max={bounds[1]}")
    print(f"Extents  {extents}")

    if not mesh.is_watertight:
        print(
            "⚠️  Mesh is NOT watertight (open edges detected).\n"
            "   Inside/outside tests may be unreliable near those regions."
        )

    return mesh