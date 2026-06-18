from ansys.mapdl.core import launch_mapdl

print("Launching MAPDL...")

mapdl = launch_mapdl()

try:
    print("Entering PREP7...")
    mapdl.prep7()

    print("Defining element...")
    mapdl.et(1, 185)

    print("Defining material...")
    mapdl.mp("EX", 1, 200e9)
    mapdl.mp("NUXY", 1, 0.3)

    print("Creating a single block...")
    mapdl.block(0, 1, 0, 1, 0, 1)

    print("Meshing block...")
    mapdl.vmesh("ALL")

    print("Switch to solve...")
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")

    # fix bottom face
    mapdl.nsel("S", "LOC", "Z", 0)
    mapdl.d("ALL", "ALL", 0)
    mapdl.allsel()

    # apply displacement on top face
    mapdl.nsel("S", "LOC", "Z", 1)
    mapdl.d("ALL", "UZ", -0.01)
    mapdl.allsel()

    print("Solving...")
    mapdl.solve()

    print("Finishing...")
    mapdl.finish()

    print("TEST COMPLETE OK")

finally:
    try:
        print("finish()...")
        mapdl.finish()
    except:
        print("finish() failed, trying exit()...")
        pass

    try:
        print("exit()...")
        mapdl.exit()
    except:
        print("exit() failed, trying _close_connection()...")
        pass

    try:
        print("_close_connection()...")
        mapdl._close_connection()
    except:
        print("_close_connection() failed, giving up.")
        pass