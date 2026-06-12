import h5py

dataset_path = "path_to_your_downloaded_file.h5"

with h5py.File(dataset_path, 'r') as f:
    print("Keys available in this HDF5 file:")
    print(list(f.keys()))
    
    for key in f.keys():
        print(f"\nKey: {key}")
        print(f"Data type: {type(f[key])}")
        print(f"Shape: {f[key].shape}")