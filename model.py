import torch
import torch.nn as nn

class LatticeCNN3D(nn.Module):
    def __init__(self):
        super(LatticeCNN3D, self).__init__()
        
        # 3D Convolutional Feature Extractor
        self.features = nn.Sequential(
            # Layer 1: Detect basic structural edges/surfaces
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), 
            
            # Layer 2: Detect local cell connectivity structures
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), 
            
            # Layer 3: Consolidate spatial features across the entire unit block
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU() 
        )
        
        # Global Average Pooling: Automatically forces spatial dimensions to 1x1x1
        # This decouples the network from rigid input resolution limitations (e.g., 20^3 or 32^3)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Regression network to output the single continuous GPa scalar value
        # input size is now strictly determined by the 64 output channels
        self.regressor = nn.Sequential(
            nn.Linear(64 * 1 * 1 * 1, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Protects model from over-memorizing training shapes
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # Single output: Predicted E_eff in GPa
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)           # Squashes remaining spatial blocks to 1x1x1
        x = x.view(x.size(0), -1) # Flattens cleanly down to [Batch Size, 64]
        x = self.regressor(x)
        return x

if __name__ == "__main__":
    model = LatticeCNN3D()
    
    # Verification Test 1: Traditional 20^3 resolution
    dummy_input_20 = torch.randn(4, 1, 20, 20, 20)
    output_20 = model(dummy_input_20)
    print("20^3 grid pass successful. Output shape:", output_20.shape)
    
    # Verification Test 2: Upgraded 32^3 resolution
    dummy_input_32 = torch.randn(4, 1, 32, 32, 32)
    output_32 = model(dummy_input_32)
    print("32^3 grid pass successful. Output shape:", output_32.shape)