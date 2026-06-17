import torch
import torch.nn as nn

class LatticeStrengthCNN3D(nn.Module):
    """Processes 64x64x64 3D grids to evaluate structural layout efficiency."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),  # -> 32x32x32
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1), # -> 16x16x16
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1), # -> 8x8x8
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool3d(1)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(64 + 1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, voxels, vfrac):
        x = self.features(voxels)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, vfrac], dim=1)
        return self.mlp(x)