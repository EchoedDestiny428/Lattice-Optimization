from ansys.mapdl.core import launch_mapdl

mapdl = launch_mapdl(
    run_location="./mapdl_run",
    override=True,
    loglevel="DEBUG"
)

print(mapdl)