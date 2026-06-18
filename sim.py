from ansys.mapdl.core import launch_mapdl
import numpy as np

BOX_SIZE = 10.0
AREA = BOX_SIZE ** 2


# -----------------------------
# METRICS
# -----------------------------
def compute_modulus(force, disp):
    strain = disp / BOX_SIZE
    stress = force / AREA

    # linear fit (small strain region)
    mask = strain < 0.02

    if np.sum(mask) < 2:
        return np.nan

    coeff = np.polyfit(strain[mask], stress[mask], 1)
    return coeff[0]


def compute_energy(force, disp):
    return np.trapz(force, disp)


# -----------------------------
# MAIN SIMULATION
# -----------------------------
def run_simulation(stl_path):

    mapdl = launch_mapdl()

    mapdl.clear()
    mapdl.prep7()

    # -------------------------
    # MATERIAL (PLA)
    # -------------------------
    E = 3e9       # Pa
    nu = 0.33

    mapdl.mp("EX", 1, E)
    mapdl.mp("NUXY", 1, nu)

    # -------------------------
    # IMPORT STL
    # -------------------------
    mapdl.stlimport(stl_path)

    mapdl.vmesh("ALL")

    # -------------------------
    # BOUNDARY CONDITIONS
    # -------------------------
    # bottom fixed
    mapdl.nsel("S", "LOC", "Z", 0)
    mapdl.d("ALL", "ALL", 0)
    mapdl.allsel()

    # top displacement
    max_disp = 0.5  # adjust strain level
    mapdl.nsel("S", "LOC", "Z", BOX_SIZE)
    mapdl.d("ALL", "UZ", -max_disp)
    mapdl.allsel()

    # -------------------------
    # SOLVE
    # -------------------------
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()

    # -------------------------
    # POST PROCESSING
    # -------------------------
    mapdl.post1()

    # reaction force at bottom
    mapdl.nsel("S", "LOC", "Z", 0)
    rf = mapdl.get_array("RF", item="FZ")

    force = np.sum(rf)

    # displacement
    disp = max_disp

    # -------------------------
    # METRICS
    # -------------------------
    E_eff = compute_modulus(force, disp)
    EA = compute_energy(np.array([force]), np.array([disp]))
    SEA = EA / (BOX_SIZE ** 3)

    mapdl.exit()

    return {
        "effective_modulus": float(E_eff),
        "peak_force": float(force),
        "ea": float(EA),
        "sea": float(SEA),
    }