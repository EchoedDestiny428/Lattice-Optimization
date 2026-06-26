import torch
import torch.nn as nn

class LatticeCNN3D(nn.Module):
    """
    Axis-Aware, Physics-Informed 3D CNN.
    Extracts geometric features, calculates global volume fraction, AND
    computes a 1D density profile along the Z-loading axis to capture 
    anisotropic load-paths and vertical pillar-effects.
    """
    def __init__(self):
        super().__init__()
        
        # Deep Spatial Feature Extractor
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # 32^3 -> 16^3

            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # 16^3 -> 8^3

            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # 8^3 -> 4^3
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)  # 4^3 -> 2^3
        )
        
        # 1024 (Spatial) + 1 (Global Vol) + 32 (Z-axis Profile) = 1057 Inputs
        self.regressor = nn.Sequential(
            nn.Linear(1057, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. PHYSICS INJECTION: Global Volume Fraction
        vol_fraction = torch.sum(x.view(batch_size, -1), dim=1, keepdim=True) / 32768.0
        
        # 2. LOAD-PATH INJECTION: Slice-by-slice density along the Z-axis (Depth)
        # We average out the Channel (1), Height (3), and Width (4) dimensions.
        # This leaves a 32-element array representing material density from top to bottom.
        z_profile = torch.mean(x, dim=[1, 3, 4]) # Shape: [Batch, 32]
        
        # Extract visual/topological features
        out = self.features(x)
        out = torch.flatten(out, 1) # Shape: [Batch, 1024]
        
        # Concatenate everything into a comprehensive structural descriptor vector
        out = torch.cat((out, vol_fraction, z_profile), dim=1)
        
        # Predict stiffness
        out = self.regressor(out)
        return out

if __name__ == "__main__":
    # Quick architecture validation pass
    test_tensor = torch.randn(2, 1, 32, 32, 32)
    net = LatticeCNN3D()
    output = net(test_tensor)
    print(f"Axis-Aware Architecture Verified! Output shape: {output.shape}")