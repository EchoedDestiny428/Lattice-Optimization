import numpy as np
import trimesh
from scipy.ndimage import label
from config import RESOLUTION, BOX_SIZE_MM

def voxelize_stl(stl_path, resolution=RESOLUTION, box_size_mm=BOX_SIZE_MM):
    mesh = trimesh.load(stl_path)

    if not mesh.is_watertight:
        print("Warning: mesh is not watertight.")
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

    max_dimension = mesh.extents.max()
    if max_dimension <= 0:
        raise ValueError("Mesh has zero size.")

    scale_factor = box_size_mm / max_dimension
    mesh.apply_scale(scale_factor)

    pitch = box_size_mm / resolution
    voxel_grid = mesh.voxelized(pitch=pitch)
    voxels = voxel_grid.matrix[:resolution, :resolution, :resolution]

    if voxels.shape != (resolution, resolution, resolution):
        padded = np.zeros((resolution, resolution, resolution), dtype=bool)
        nx, ny, nz = voxels.shape
        
        # Center the lattice within the padded bounding box
        ox = (resolution - nx) // 2
        oy = (resolution - ny) // 2
        oz = (resolution - nz) // 2
        
        padded[ox:ox+nx, oy:oy+ny, oz:oz+nz] = voxels
        voxels = padded

    # Largest 6-connected component cleanup
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1

    labels_array, n = label(voxels, structure=structure)
    if n > 0:
        sizes = np.bincount(labels_array.ravel())
        sizes[0] = 0
        largest = np.argmax(sizes)
        voxels = labels_array == largest

    density = float(voxels.mean())
    return voxels, density