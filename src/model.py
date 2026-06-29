import torch
import torch.nn as nn
from config import RESOLUTION

class LatticeCNN3D(nn.Module):
    def __init__(self, resolution=RESOLUTION):
        super().__init__()
        self.resolution = resolution

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, resolution, resolution, resolution)
            out = self.features(dummy)
            self.feature_size = out.view(1, -1).shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(self.feature_size + 1 + resolution, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        b = x.size(0)

        vol_fraction = torch.sum(x.view(b, -1), dim=1, keepdim=True) / (self.resolution ** 3)
        
        # 🚨 FIX: Average across Channel(1), X(2), and Y(3) to isolate the Z-axis profile
        z_profile = torch.mean(x, dim=[1, 2, 3])  

        out = self.features(x)
        out = torch.flatten(out, 1)
        out = torch.cat((out, vol_fraction, z_profile), dim=1)

        return self.regressor(out)