import numpy as np
import trimesh
from scipy.ndimage import label


def voxelize_stl(stl_path, resolution=32, box_size_mm=10.0):
    """
    Convert an STL mesh into a fixed-resolution voxel grid.

    Pipeline:
        STL
        -> scale to box_size_mm
        -> voxelize
        -> crop/pad
        -> keep largest 6-connected component
        -> compute density

    Returns
    -------
    voxels : np.ndarray
        Boolean array of shape (resolution, resolution, resolution)

    density : float
        Fraction of occupied voxels.
    """

    # --------------------------------------------------------
    # Load mesh
    # --------------------------------------------------------

    mesh = trimesh.load(stl_path)

    if not mesh.is_watertight:
        print("Warning: mesh is not watertight.")
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

    # --------------------------------------------------------
    # Scale to standard box
    # --------------------------------------------------------

    max_dimension = mesh.extents.max()

    if max_dimension <= 0:
        raise ValueError("Mesh has zero size.")

    scale_factor = box_size_mm / max_dimension
    mesh.apply_scale(scale_factor)

    # --------------------------------------------------------
    # Voxelization
    # --------------------------------------------------------

    pitch = box_size_mm / resolution

    voxel_grid = mesh.voxelized(pitch=pitch)

    voxels = voxel_grid.matrix[:resolution, :resolution, :resolution]

    # --------------------------------------------------------
    # Pad to fixed resolution
    # --------------------------------------------------------

    if voxels.shape != (resolution, resolution, resolution):

        padded = np.zeros(
            (resolution, resolution, resolution),
            dtype=bool,
        )

        nx, ny, nz = voxels.shape

        padded[:nx, :ny, :nz] = voxels

        voxels = padded

    # --------------------------------------------------------
    # Largest 6-connected component
    # --------------------------------------------------------

    structure = np.zeros((3, 3, 3), dtype=int)

    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1

    labels, n = label(voxels, structure=structure)

    if n > 0:

        sizes = np.bincount(labels.ravel())

        sizes[0] = 0

        largest = np.argmax(sizes)

        voxels = labels == largest

    density = float(voxels.mean())

    return voxels, density