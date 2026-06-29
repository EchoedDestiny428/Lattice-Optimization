import os
import torch

# 1. ROOT DIRECTORY (The "Base" anchor)
# This finds where config.py is located and sets that as the root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. DEVICE CONFIGURATION
# Set this once here and import it everywhere to avoid code duplication
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. GEOMETRY / LATTICE SETTINGS
RESOLUTION = 32
BOX_SIZE_MM = 10.0
NUM_SAMPLES = 100

# 4. PATHS (All built relative to BASE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(DATASET_DIR, "samples")
STL_DIR = os.path.join(DATASET_DIR, "generated_stl")
CSV_PATH = os.path.join(DATASET_DIR, "dataset_compiled.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_WEIGHTS_PATH = os.path.join(MODELS_DIR, "lattice_cnn.pth")

# 5. LEGACY / GENERATION SETTINGS
# Since you are moving to procedural/harmonic generation, 
# you can eventually remove these or keep them for 'baseline' reference.
SHAPES = ["gyroid", "primitive", "diamond", "i_wp", "neovius"]