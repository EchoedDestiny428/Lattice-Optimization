import torch
import torch.nn as nn

class LatticeCNN3D(nn.Module):
    """
    Physics-Informed 3D CNN.
    Extracts spatial features via convolutions, but explicitly calculates 
    Volume Fraction and injects it into the final regression layers to anchor 
    predictions in real-world Gibson-Ashby scaling laws.
    """
    def __init__(self):
        super().__init__()
        
        # Deep Spatial Feature Extractor
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # Downsamples 32^3 -> 16^3

            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # Downsamples 16^3 -> 8^3

            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # Downsamples 8^3 -> 4^3
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)  # Downsamples 4^3 -> 2^3
        )
        
        # 128 channels * (2 * 2 * 2) spatial map = 1024 spatial features
        # PLUS 1 explicit Physics Feature (Volume Fraction) = 1025 features
        self.regressor = nn.Sequential(
            nn.Linear(1024 + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # ============================================================
        # PHYSICS INJECTION: Calculate exact volume fraction 
        # (Total solid voxels divided by 32768 total volume)
        # ============================================================
        vol_fraction = torch.sum(x.view(batch_size, -1), dim=1, keepdim=True) / 32768.0
        
        # Extract spatial/topological features
        out = self.features(x)
        out = torch.flatten(out, 1)
        
        # Concatenate the hard physics variable with the learned spatial variables
        out = torch.cat((out, vol_fraction), dim=1)
        
        # Predict stiffness based on combined data
        out = self.regressor(out)
        return out

if __name__ == "__main__":
    # Sanity Check
    test_tensor = torch.randn(2, 1, 32, 32, 32)
    net = LatticeCNN3D()
    output = net(test_tensor)
    print(f"Physics-Informed Check Passed! Output shape: {output.shape}")