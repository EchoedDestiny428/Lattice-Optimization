import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import trimesh
import numpy as np
from tqdm import tqdm
from matplotlib.path import Path

# 1. NETWORKING ARCHITECTURE
class Lattice3DCNN(nn.Module):
    def __init__(self):
        super(Lattice3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten() 
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 8 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.regressor(self.features(x))

# 2. RAW H5 DATA PIPELINE
class GLU3DDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            self.dataset_length = f['voxels'].shape[0]
        self.scale_factor = 1e7  

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            voxel_grid = f['voxels'][idx].astype(np.float32)
            stiffness_matrix = f['stiffness'][idx]
            true_score = stiffness_matrix[0, 0] / self.scale_factor 

        return torch.tensor(voxel_grid).unsqueeze(0), torch.tensor(true_score).unsqueeze(0)

# 3. VERIFIED 0.0 TO 1.0 VOXELIZER
def stl_to_voxel_tensor(stl_path, grid_size=64, save_debug_h5=None):
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Target STL mesh not found at: {stl_path}")
        
    mesh = trimesh.load(stl_path)
    stl_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    print("\n[PROCESSING] Running Zero-RAM Vectorized Vertex Mapping...")
    
    vertices = mesh.vertices
    if len(vertices) == 0:
        print("[WARNING] No vertices found in STL. Returning empty grid.")
        return torch.zeros((1, grid_size, grid_size, grid_size), dtype=torch.float32)
    
    # Scale coordinates from physical space [0.0, 1.0] to matrix index space [0, 63]
    scaled_indices = np.floor((vertices - 1e-5) * grid_size).astype(np.int32)
    scaled_indices = np.clip(scaled_indices, 0, grid_size - 1)
    
    # Chunk the vertices to give the progress bar steps to iterate through
    # Using 10 chunks keeps the loop overhead low while showing a smooth status bar
    chunk_size = max(1, len(scaled_indices) // 10)
    
    for i in tqdm(range(0, len(scaled_indices), chunk_size), desc="Mapping Vertices", unit="chunk"):
        chunk = scaled_indices[i : i + chunk_size]
        stl_matrix[chunk[:, 0], chunk[:, 1], chunk[:, 2]] = 1.0
        
    print(f"[INFO] Voxelization complete. Total Solid Voxels Detected: {int(np.sum(stl_matrix))}")
    
    if save_debug_h5:
        output_dir = os.path.dirname(save_debug_h5)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with h5py.File(save_debug_h5, 'w') as f:
            f.create_dataset('voxels', data=np.expand_dims(stl_matrix, axis=0), dtype='f4')
        print(f"[DEBUG SUCCESS] Reconstructed voxel array written to: '{save_debug_h5}'")

    return torch.tensor(stl_matrix).unsqueeze(0)



# 4. EXECUTION LOOP
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sample_h5 = os.path.join("data", "h5_files", "test2.h5")
    sample_stl = os.path.join("data", "stl_files", "standardized_lattice.stl")
    
    # Setup final destination path
    model_dir = os.path.join("data", "models")
    os.makedirs(model_dir, exist_ok=True)
    final_weights_path = os.path.join(model_dir, "lattice_3dcnn_final.pth")
    
    # User Menu Selection
    print("=============================================")
    print("   Lattice 3D CNN Training Interface         ")
    print("=============================================")
    print("1) Load previously saved model weights")
    print("2) Train a brand new model from scratch")
    choice = input("\nSelect an option (1 or 2): ").strip()
    
    should_train = True
    model = Lattice3DCNN().to(device)
    
    if choice == '1':
        if os.path.exists(final_weights_path):
            print(f"\n[INFO] Found existing weights at: {final_weights_path}")
            model.load_state_dict(torch.load(final_weights_path, map_location=device))
            print("[SUCCESS] Pre-trained weights successfully loaded into model structure.")
            should_train = False
        else:
            print(f"\n[WARNING] Weights file not found at '{final_weights_path}'.")
            print("Defaulting back to training mode...")
    
    dataset = GLU3DDataset(sample_h5)
    
    # 5. OPTIONAL TRAINING BLOCK
    if should_train:
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("\nBeginning network training loop...")
        for epoch in range(15):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(dataset)
            print(f"Epoch [{epoch+1:02d}/15] Complete | Loss: {epoch_loss:.6f}")
            
        # ONLY SAVE HERE: Once training loop ends completely
        torch.save(model.state_dict(), final_weights_path)
        print(f"\n[SUCCESS] Training finished. Weights secured at: {final_weights_path}")
    
    # 6. VALIDATION TESTING RUN
    print("\nPreparing model evaluation pass...")
    model.eval()

    test_idx = 550
    
    with torch.no_grad():
        with h5py.File(sample_h5, 'r') as f:
            true_h5_val = f['stiffness'][test_idx][0, 0]
            
        h5_input = dataset[test_idx][0].unsqueeze(0).to(device)
        pred_h5 = model(h5_input).item() * dataset.scale_factor
        print(f"\nH5 Target Baseline (Index {test_idx}): {true_h5_val:.4f} | H5 Model Prediction: {pred_h5:.4f}")
        
        if os.path.exists(sample_stl):
            debug_h5_output = os.path.join("data", "h5_files", "debug.h5")

            print(f"Voxelizing and running inference on: {sample_stl}")
            stl_input = stl_to_voxel_tensor(sample_stl, save_debug_h5=debug_h5_output).unsqueeze(0).to(device)
            pred_stl = model(stl_input).item() * dataset.scale_factor
            print(f"STL Model Prediction:  {pred_stl:.4f}")
        else:
            print(f"[ERROR] Verification STL missing at {sample_stl}. Skipping target check.")