"""
tester.py
---------
Evaluates whether generated subgrid coordinate points reside inside
the bounding coordinates of the specified 3D triangle mesh.
"""

import numpy as np
import trimesh


def test_points_inside(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    batch_size: int = 500_000,
) -> np.ndarray:
    """
    Test points inside the mesh in memory-capped batched blocks.
    """
    total = len(points)
    inside = np.empty(total, dtype=bool)

    print("Testing inside/outside …")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        inside[start:end] = mesh.contains(points[start:end])
        pct = end / total * 100
        print(f"  {pct:5.1f}%", end="\r", flush=True)

    print(" 100.0%  done ")
    return inside