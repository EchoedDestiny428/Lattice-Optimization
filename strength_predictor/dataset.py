import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

def compute_voigt_bulk_modulus(stiffness_tensors: np.ndarray) -> np.ndarray:
    C = stiffness_tensors
    C11, C22, C33 = C[:, 0, 0], C[:, 1, 1], C[:, 2, 2]
    C12, C13, C23 = C[:, 0, 1], C[:, 0, 2], C[:, 1, 2]
    return (C11 + C22 + C33 + 2.0 * (C12 + C13 + C23)) / 9.0

class MinMaxNormalizer:
    def __init__(self, min_val, max_val):
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.range = max(self.max_val - self.min_val, 1e-8)

    def transform(self, array):
        return (array - self.min_val) / self.range

class GLU3DLatticeDataset(Dataset):
    def __init__(self, h5_path: str, normalizer=None, augment=False):
        self.augment = augment
        print(f"Opening dataset: {h5_path}")
        
        with h5py.File(h5_path, "r") as f:
            self.voxels = f["voxels"][()].astype(np.float32)
            stiffness = f["stiffness"][()]
            self.vfrac = f["vfrac"][()].astype(np.float32)

        K_values = compute_voigt_bulk_modulus(stiffness)

        if normalizer is None:
            self.normalizer = MinMaxNormalizer(K_values.min(), K_values.max())
            print(f"Locked Normalization -> Min: {K_values.min():.2f}, Max: {K_values.max():.2f}")
        else:
            self.normalizer = normalizer

        self.labels = self.normalizer.transform(K_values).astype(np.float32)

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        voxel = self.voxels[idx]
        vfrac = self.vfrac[idx]
        label = self.labels[idx]

        if self.augment:
            if np.random.rand() > 0.5: voxel = np.flip(voxel, axis=0)
            if np.random.rand() > 0.5: voxel = np.flip(voxel, axis=1)

        voxel_tensor = torch.FloatTensor(voxel.copy()).unsqueeze(0)
        return voxel_tensor, torch.FloatTensor([vfrac]), torch.FloatTensor([label])