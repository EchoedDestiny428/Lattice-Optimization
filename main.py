from sim import run_simulation
import os
import json

stl_path = os.path.join("data", "samples", "sample_000000", "lattice.stl")

labels = run_simulation(stl_path)

with open("labels.json", "w") as f:
    json.dump(labels, f, indent=4)