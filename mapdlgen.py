from ansys.mapdl.core import launch_mapdl
import numpy as np

BOX_SIZE = 10.0


def voxels_to_mapdl(voxels):

    nx, ny, nz = voxels.shape

    dx = BOX_SIZE / nx
    dy = BOX_SIZE / ny
    dz = BOX_SIZE / nz

    print("Launching MAPDL...")

    mapdl = launch_mapdl()

    mapdl.clear()
    mapdl.prep7()

    # ------------------
    # Element + material
    # ------------------
    mapdl.et(1, 185)

    mapdl.mp("EX", 1, 200e9)
    mapdl.mp("NUXY", 1, 0.3)

    # ------------------
    # Node lookup table
    # ------------------
    node_map = {}

    next_node = 1

    def get_node(i, j, k):

        nonlocal next_node

        key = (i, j, k)

        if key in node_map:
            return node_map[key]

        x = i * dx
        y = j * dy
        z = k * dz

        mapdl.n(next_node, x, y, z)

        node_map[key] = next_node

        next_node += 1

        return node_map[key]

    # ------------------
    # Create elements
    # ------------------
    element_count = 0

    print("Creating voxel mesh...")

    for i in range(nx):

        for j in range(ny):

            for k in range(nz):

                if voxels[i, j, k] == 0:
                    continue

                n1 = get_node(i,   j,   k)
                n2 = get_node(i+1, j,   k)
                n3 = get_node(i+1, j+1, k)
                n4 = get_node(i,   j+1, k)

                n5 = get_node(i,   j,   k+1)
                n6 = get_node(i+1, j,   k+1)
                n7 = get_node(i+1, j+1, k+1)
                n8 = get_node(i,   j+1, k+1)

                mapdl.e(
                    n1, n2, n3, n4,
                    n5, n6, n7, n8
                )

                element_count += 1

    print()
    print("Voxel mesh created")
    print("Solid voxels :", int(voxels.sum()))
    print("Nodes        :", len(node_map))
    print("Elements     :", element_count)

    return mapdl