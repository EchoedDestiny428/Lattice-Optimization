from pathlib import Path
import torch

# 1. ROOT DIRECTORY
# This makes ROOT_DIR the absolute path to your project folder
ROOT_DIR = Path(__file__).resolve().parent

# 2. DEVICE CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. DIRECTORY STRUCTURE
DATASET_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATASET_DIR / "samples"
STL_DIR = DATASET_DIR / "generated_stl"
MODELS_DIR = ROOT_DIR / "models"

# 4. FILE PATHS
CSV_PATH = DATASET_DIR / "dataset_compiled.csv"
MODEL_WEIGHTS_PATH = MODELS_DIR / "lattice_cnn.pth"

# 5. GENERATION & TRAINING SETTINGS
RESOLUTION = 48
BOX_SIZE_MM = 10.0
NUM_SAMPLES = 10000
BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE = 3e-4

# Ensure critical directories exist automatically
for folder in [DATASET_DIR, SAMPLES_DIR, STL_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)