import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from config import CSV_PATH, DATASET_DIR, RESOLUTION

class VoxelLatticeDataset(Dataset):
    def __init__(self, csv_path=CSV_PATH, samples_dir=DATASET_DIR, resolution=RESOLUTION):
        df = pd.read_csv(csv_path)
        self.df = df[(df["status"] == "SUCCESS") & (df["E_eff_gpa"] > 0.001)].reset_index(drop=True)
        self.samples_dir = samples_dir
        self.resolution = resolution
        print(f"[Dataset] Loaded {len(self.df)} valid samples (filtered).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]

        voxel_path = os.path.join(self.samples_dir, sample_id, "voxels.npz")
        data = np.load(voxel_path)
        voxels = data["voxels"]

        if voxels.shape != (self.resolution, self.resolution, self.resolution):
            raise ValueError(f"Voxel shape mismatch for {sample_id}: got {voxels.shape}, expected {(self.resolution,) * 3}")

        voxel_tensor = torch.from_numpy(voxels.astype(np.float32)).unsqueeze(0)  # (1, D, H, W)
        target = torch.tensor(row["E_eff_gpa"], dtype=torch.float32).unsqueeze(0)

        return voxel_tensor, target