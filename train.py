# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.models import LatticeEfficiencyNet
from src.generator import generate_lattice

# ==============================================================================
USE_SYNTHETIC = True  
# ==============================================================================

class RealLatticeDataset(Dataset):
    """Loads true engineering data from an Excel/CSV file and saved .npy meshes."""
    def __init__(self, csv_file, voxel_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.voxel_dir = voxel_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        voxel_filename = self.data_frame.iloc[idx]['mesh_filename']
        true_efficiency = self.data_frame.iloc[idx]['true_efficiency_factor']
        
        voxel_path = os.path.join(self.voxel_dir, voxel_filename)
        voxels = np.load(voxel_path)
        
        voxel_tensor = torch.tensor(voxels, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor([true_efficiency], dtype=torch.float32)
        return voxel_tensor, target_tensor


def generate_synthetic_batch(batch_size=2):
    """Generates on-the-fly math grids for safe pipeline testing."""
    inputs, targets = [], []
    for _ in range(batch_size):
        l_type = np.random.choice(['gyroid', 'schwarz_p'])
        random_percent = np.random.uniform(15.0, 50.0)
        
        voxels, _ = generate_lattice(equation_type=l_type, target_percent=random_percent)
        fake_true_score = 1.10 if l_type == 'gyroid' else 0.90 # gyroid's just superior in general
        fake_true_score += np.random.uniform(-0.05, 0.05)
        
        inputs.append(np.expand_dims(voxels, axis=0))
        targets.append([fake_true_score])
        
    return torch.tensor(np.array(inputs), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)


def train_model():
    print("==================================================")
    print("        LATTICE AI TRAINING INFRASTRUCTURE        ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Computation Engine: {device}")
    
    # 1. Initialize Network Architecture
    model = LatticeEfficiencyNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    batch_size = 2
    
    # 2. Handle Data Source Routing based on the Toggle Switch
    if USE_SYNTHETIC:
        print("\n[DATA MODE] ---> SYNTHETIC SANDBOX ACTIVE <---")
        print("Generating mock math shapes automatically for validation...")
    else:
        print("\n[DATA MODE] ---> REAL PRODUCTION DATA ACTIVE <---")
        CSV_PATH = "data/lab_results.csv"
        VOXEL_FOLDER = "data/voxel_grids/"
        
        if not os.path.exists(CSV_PATH):
            print(f"[ERROR] Missing '{CSV_PATH}'. Switch back to USE_SYNTHETIC = True to test.")
            return
            
        dataset = RealLatticeDataset(csv_file=CSV_PATH, voxel_dir=VOXEL_FOLDER)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Main Optimization Loop
    print("\n[TRAINING] Starting neural optimization steps...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        if USE_SYNTHETIC:
            # Synthetic Loop: Fresh mathematical shapes every single epoch step
            inputs, targets = generate_synthetic_batch(batch_size=batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
        else:
            # Production Loop: Pulling your true lab data batches smoothly
            batch_count = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
            running_loss /= max(1, batch_count)
            
        print(f"Epoch [{epoch+1}/{epochs}] ───► Average Loss: {running_loss:.6f}")
        
    # 4. Save Finalized Weight File
    torch.save(model.state_dict(), "lattice_efficiency_model.pth")
    print("\n[SUCCESS] Pipeline checked! Model saved to 'lattice_efficiency_model.pth'")

if __name__ == "__main__":
    train_model()