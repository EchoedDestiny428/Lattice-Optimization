import torch
import torch.nn as nn
from config import RESOLUTION

class LatticeCNN3D(nn.Module):
    def __init__(self, resolution=48):
        super(LatticeCNN3D, self).__init__()
        self.resolution = resolution

        # Use stride=2 convolutions instead of Pooling to learn how to preserve mass/density
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, bias=False), # 48 -> 24
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 24 -> 12
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 12 -> 6
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 6 -> 3
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Dropout(0.5), # Increased Dropout to prevent memorization
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out = self.features(x)
        # Global Average Pooling across all 3 dimensions
        out = torch.mean(out, dim=[2, 3, 4]) 
        return self.regressor(out)