import numpy as np
from ansys.mapdl.core import launch_mapdl


def voxels_to_mapdl(voxels, box_size=10.0):

    from ansys.mapdl.core import launch_mapdl
    mapdl = launch_mapdl()

    nx, ny, nz = voxels.shape

    pitch = box_size / nx

    mapdl.clear()
    mapdl.prep7()

    mapdl.et(1, 185)

    mapdl.block(0, box_size, 0, box_size, 0, box_size)

    mapdl.esize(pitch)
    mapdl.vmesh("ALL")

    return mapdl