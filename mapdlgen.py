import os
import numpy as np


def voxels_to_mapdl(mapdl, voxels, box_size=0.01):
    """Blazing fast voxel meshing using direct workspace file input."""
    mapdl.clear()
    mapdl.prep7()

    nx, ny, nz = voxels.shape
    dx = box_size / nx
    dy = box_size / ny
    dz = box_size / nz

    mapdl.et(1, 185)  # SOLID185
    mapdl.mp("EX", 1, 200e9)
    mapdl.mp("NUXY", 1, 0.3)

    node_map = {}
    next_node = 1

    # Save directly inside MAPDL's own active working directory
    inp_filename = "temp_mesh.inp"
    full_inp_path = os.path.join(mapdl.directory, inp_filename)

    with open(full_inp_path, "w") as f:
        # 1. Pre-generate all grid bounding nodes via fast text stream
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    x, y, z = i * dx, j * dy, k * dz
                    f.write(f"N,{next_node},{x},{y},{z}\n")
                    node_map[(i, j, k)] = next_node
                    next_node += 1

        # 2. Write elements
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if voxels[i, j, k] == 0:
                        continue

                    n1 = node_map[(i, j, k)]
                    n2 = node_map[(i + 1, j, k)]
                    n3 = node_map[(i + 1, j + 1, k)]
                    n4 = node_map[(i, j + 1, k)]
                    n5 = node_map[(i, j, k + 1)]
                    n6 = node_map[(i + 1, j, k + 1)]
                    n7 = node_map[(i + 1, j + 1, k + 1)]
                    n8 = node_map[(i, j + 1, k + 1)]

                    f.write(f"E,{n1},{n2},{n3},{n4},{n5},{n6},{n7},{n8}\n")

    # Read the file natively from its own directory
    mapdl.input(inp_filename)
    
    # Optional cleanup: remove the file so 1,000 runs don't clutter the disk
    try:
        os.remove(full_inp_path)
    except:
        pass

    print(f"Voxel block mesh loaded instantly via file stream.")
    return mapdl