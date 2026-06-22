from ansys.mapdl.core import launch_mapdl
import numpy as np

BOX_SIZE = 10.0


def voxels_to_mapdl(voxels):

    nx, ny, nz = voxels.shape

    mapdl = launch_mapdl()
    mapdl.clear()
    mapdl.prep7()

    # element + material
    mapdl.et(1, 185)
    mapdl.mp("EX", 1, 200e9)
    mapdl.mp("NUXY", 1, 0.3)

    # create block ONLY (fast)
    mapdl.block(0, BOX_SIZE, 0, BOX_SIZE, 0, BOX_SIZE)

    # mesh block
    mapdl.esize(BOX_SIZE / nx)
    mapdl.vmesh("ALL")

    print(f"Created mesh from block")

    return mapdl