"""
stl_voxelizer
~~~~~~~~~~~~~
Convert STL files to 3-D voxel grids with 0–1 fill values.

Quick start
-----------
>>> from stl_voxelizer import stl_to_voxels
>>> grid = stl_to_voxels("model.stl", resolution=64)
>>> print(grid.shape)   # e.g. (64, 48, 32)
"""

from .voxelize import stl_to_voxels

__all__ = ["stl_to_voxels"]