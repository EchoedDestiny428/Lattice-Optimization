from scripts import step01_generate, step02_export_stl, step03_voxelize_stl, step04_simulate, step05_train_cnn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

def main():
    step01_generate.generate_voxel_samples()

    step02_export_stl.voxels_to_stl()

    step03_voxelize_stl.run_voxelization()

    step04_simulate.simulate()

    step05_train_cnn.train_cnn()
    
if __name__ == "__main__":
    main()