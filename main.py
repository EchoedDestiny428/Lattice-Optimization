from scripts import step01_generate, step02_export_stl, step03_voxelize_stl, step04_simulate, step05_train_cnn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

def main():
    step01_generate.generate_voxel_samples()

    
    try:
        user_input = input("Enter number of samples to process: ")
        step02_export_stl.voxels_to_stl(int(user_input))
    except ValueError:
        print("Please enter a valid integer.")

    step04_simulate.simulate()

    
    # step05_train_cnn.train_cnn()
    
if __name__ == "__main__":
    main()