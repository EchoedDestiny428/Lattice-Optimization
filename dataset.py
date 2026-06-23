import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class VoxelLatticeDataset(Dataset):
    def __init__(self, csv_path, samples_dir):
        """
        Custom PyTorch Dataset for loading 3D voxel matrices alongside physical targets.
        """
        df = pd.read_csv(csv_path)
        
        self.df = df[(df["status"] == "SUCCESS") & (df["E_eff_gpa"] > 0.05)].reset_index(drop=True)
        self.samples_dir = samples_dir
        
        print(f"[Dataset] Filtered raw records down to {len(self.df)} clean, load-bearing samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        
        voxel_path = os.path.join(self.samples_dir, sample_id, "voxels.npz")
        voxels = np.load(voxel_path)["voxels"] # Shape: (20, 20, 20)
        
        voxel_tensor = torch.tensor(voxels, dtype=torch.float32).unsqueeze(0)
        
        target_gpa = torch.tensor(row["E_eff_gpa"], dtype=torch.float32).unsqueeze(0)
        
        return voxel_tensor, target_gpa

if __name__ == "__main__":
    ds = VoxelLatticeDataset("data/dataset_compiled.csv", "data/samples")
    if len(ds) > 0:
        v, t = ds[0]
        print(f"Success! Voxel tensor shape: {v.shape} | Target stiffness shape: {t.shape}")