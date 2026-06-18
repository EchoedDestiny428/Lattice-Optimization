from ansys.mapdl.core import launch_mapdl

BOX_SIZE = 10.0


def voxels_to_mapdl(voxels):

    nx, ny, nz = voxels.shape
    dx = BOX_SIZE / nx

    print("Launching MAPDL...")

    mapdl = launch_mapdl()

    mapdl.clear()
    mapdl.prep7()

    print("Defining element/material...")

    mapdl.et(1, 185)

    mapdl.mp("EX", 1, 200e9)
    mapdl.mp("NUXY", 1, 0.3)

    print("Creating nodes...")

    node_map = {}
    nid = 1

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):

                mapdl.n(nid, i * dx, j * dx, k * dx)

                node_map[(i, j, k)] = nid
                nid += 1

    print("Creating elements...")

    elem_count = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                if not voxels[i, j, k]:
                    continue

                n = node_map

                mapdl.e(
                    n[(i, j, k)],
                    n[(i + 1, j, k)],
                    n[(i + 1, j + 1, k)],
                    n[(i, j + 1, k)],
                    n[(i, j, k + 1)],
                    n[(i + 1, j, k + 1)],
                    n[(i + 1, j + 1, k + 1)],
                    n[(i, j + 1, k + 1)],
                )

                elem_count += 1

    print(f"Created {elem_count} elements")

    return mapdl