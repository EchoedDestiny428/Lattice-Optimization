from ansys.mapdl.core import launch_mapdl
import numpy as np


def voxels_to_mapdl(voxels, box_size=0.01):
    """Launches MAPDL and creates a solid hexahedral mesh from a voxel matrix.

    Args:
        voxels (np.ndarray): 3D binary array indicating solid voxels.
        box_size (float): The target physical length of the RVE cube side (in
          meters). Default is 0.01 m (10 mm).
    """
    nx, ny, nz = voxels.shape

    # Define physical element sizes based on provided box_size
    dx = box_size / nx
    dy = box_size / ny
    dz = box_size / nz

    print("Launching MAPDL...")
    mapdl = launch_mapdl()

    mapdl.clear()
    mapdl.prep7()

    # ------------------
    # Element + Material
    # ------------------
    mapdl.et(1, 185)  # SOLID185 8-node structural solid

    mapdl.mp("EX", 1, 200e9)  # 200 GPa (N/m^2)
    mapdl.mp("NUXY", 1, 0.3)  # Poisson's ratio

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

                # Define the 8 corner vertices of the hexahedron
                n1 = get_node(i, j, k)
                n2 = get_node(i + 1, j, k)
                n3 = get_node(i + 1, j + 1, k)
                n4 = get_node(i, j + 1, k)

                n5 = get_node(i, j, k + 1)
                n6 = get_node(i + 1, j, k + 1)
                n7 = get_node(i + 1, j + 1, k + 1)
                n8 = get_node(i, j + 1, k + 1)

                mapdl.e(n1, n2, n3, n4, n5, n6, n7, n8)
                element_count += 1

    print()
    print("Voxel mesh created")
    print("Solid voxels :", int(voxels.sum()))
    print("Nodes        :", len(node_map))
    print("Elements     :", element_count)

    return mapdl