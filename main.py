import os
import csv
import ansys.mapdl.core as pymapdl

STL_DIR = os.path.join("data", "generated_stl")
OUTPUT_CSV = os.path.join("data", "lattice_elastic_strength.csv")

mapdl = pymapdl.launch_mapdl()

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "relative_strength"])

    for file in sorted(os.listdir(STL_DIR)):
        if not file.endswith(".stl"):
            continue
            
        stl_path = os.path.abspath(os.path.join(STL_DIR, file))
        print(f"Simulating elastic limit for {file}...")
        
        mapdl.clear()
        mapdl.prep7()
        mapdl.readdsp(stl_path)
        
        mapdl.mp("EX", 1, 3500)    
        mapdl.mp("NUXY", 1, 0.35)  
        
        mapdl.et(1, 187) 
        mapdl.esize(0.5) 
        mapdl.vmesh("ALL")
        
        mapdl.nsel("S", "LOC", "Z", 0, 0.1)
        mapdl.d("ALL", "ALL", 0)
        
        mapdl.nsel("S", "LOC", "Z", 9.9, 10.0)
        mapdl.d("ALL", "UZ", -0.2)
        mapdl.allsel()
        
        mapdl.run("/SOLU")
        mapdl.solve()
        mapdl.finish()
        
        mapdl.run("/POST1")
        mapdl.nsel("S", "LOC", "Z", 9.9, 10.0)
        mapdl.fsum()
        
        f_total = mapdl.get_value("FSUM", 0, "ITEM", "ZF") 
        f_lattice_peak = abs(f_total)
        
        f_solid_peak = 7000.0 
        relative_strength = f_lattice_peak / f_solid_peak
        
        writer.writerow([file, round(relative_strength, 5)])
        print(f"{file} | Relative Elastic Strength: {round(relative_strength * 100, 2)}%")

mapdl.exit()
print(f"\nAll data successfully saved to {OUTPUT_CSV}")