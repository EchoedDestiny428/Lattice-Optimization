import torch
import torch.nn as nn

class LatticeCNN3D(nn.Module):
    def __init__(self):
        super(LatticeCNN3D, self).__init__()
        
        # Input tensor size: (1, 20, 20, 20)
        self.features = nn.Sequential(
            # Layer 1: Detect basic structural edges/surfaces
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output dimension: (16, 10, 10, 10)
            
            # Layer 2: Detect local cell connectivity structures
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output dimension: (32, 5, 5, 5)
            
            # Layer 3: Consolidate spatial features across the entire unit block
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()                             # Output dimension: (64, 5, 5, 5)
        )
        
        # Regression network to output the single continuous GPa scalar value
        # 64 channels * 5 * 5 * 5 spatial grid = 8000 flattened nodes
        self.regressor = nn.Sequential(
            nn.Linear(64 * 5 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Protects model from over-memorizing training shapes
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # Single output: Predicted E_eff in GPa
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten spatial representation into a vector
        x = self.regressor(x)
        return x

if __name__ == "__main__":
    # Test pass with dummy array mimicking our dataset batch
    model = LatticeCNN3D()
    dummy_input = torch.randn(4, 1, 20, 20, 20) # Batch size of 4
    output = model(dummy_input)
    print("Model functional. Forward output tensor shape:", output.shape)