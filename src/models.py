# src/models.py
import torch
import torch.nn as nn

class LatticeEfficiencyNet(nn.Module):
    def __init__(self):
        super(LatticeEfficiencyNet, self).__init__()
        
        # Input tensor shape: (Batch_Size, 1, 64, 64, 64) 
        self.features = nn.Sequential(
            # Layer 1: Downsample from 64x64x64 to 32x32x32
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            
            # Layer 2: Downsample from 32x32x32 to 16x16x16
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            # Layer 3: Downsample from 16x16x16 to 8x8x8
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            # Compress the final 8x8x8 spatial blocks into a flat 64-feature vector
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Regression network that turns spatial features into efficiency factor
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # Softplus maps output to strictly > 0.0, preventing negative scaling
            nn.Softplus() 
        )
        
    def forward(self, x):
        # Extract and flatten features across the 3D voxel lattice
        x = self.features(x)
        x = torch.flatten(x, 1)

        # Calculate the final geometric efficiency factor
        efficiency_factor = self.regressor(x)
        return efficiency_factor