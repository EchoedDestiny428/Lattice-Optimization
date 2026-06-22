import numpy as np


def voxels_to_mapdl(mapdl, voxels, box_size=0.01):
    """Uses an existing active MAPDL instance to create a solid hexahedral mesh.

    Args:
        mapdl: The active PyMAPDL launch instance object.
        voxels (np.ndarray): 3D binary array indicating solid voxels.
        box_size (float): Target length of the RVE cube side (in meters).
    """
    # CRITICAL: Clear the previous iteration's geometry/solution from memory
    mapdl.clear()
    mapdl.prep7()

    nx, ny, nz = voxels.shape
    dx = box_size / nx
    dy = box_size / ny
    dz = box_size / nz

    # Element + Material
    mapdl.et(1, 185)  # SOLID185
    mapdl.mp("EX", 1, 200e9)
    mapdl.mp("NUXY", 1, 0.3)

    # Node lookup table
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

    # Create elements
    element_count = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxels[i, j, k] == 0:
                    continue

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

    print(f"Voxel mesh rebuilt | Nodes: {len(node_map)} | Elements: {element_count}")
    return mapdl