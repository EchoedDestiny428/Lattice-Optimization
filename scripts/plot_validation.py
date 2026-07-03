import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import VoxelLatticeDataset
from src.model import LatticeCNN3D
from config import CSV_PATH, SAMPLES_DIR, MODEL_WEIGHTS_PATH, DEVICE, BATCH_SIZE

def plot_validation():
    print("Loading dataset...")
    full_dataset = VoxelLatticeDataset(csv_path=CSV_PATH, samples_dir=SAMPLES_DIR)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Must use the exact same seed to get the same validation set
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Loading model...")
    model = LatticeCNN3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    all_targets = []
    all_preds = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for voxel_batch, target_batch in val_loader:
            voxel_batch = voxel_batch.to(DEVICE)
            preds = model(voxel_batch)
            
            # Convert back from log10 scale to actual GPa
            all_targets.extend((10 ** target_batch.cpu().numpy()).flatten())
            all_preds.extend((10 ** preds.cpu().numpy()).flatten())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    # Calculate error metrics
    errors = np.abs(all_targets - all_preds) / all_targets * 100
    median_error = np.median(errors)
    mean_error = np.mean(errors)
    
    print(f"\nValidation Set Statistics ({len(all_targets)} samples):")
    print(f"Median Error: {median_error:.2f}%")
    print(f"Mean Error:   {mean_error:.2f}%")
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.5, color='blue', label='Predictions')
    
    # Perfect prediction line
    max_val = max(np.max(all_targets), np.max(all_preds))
    min_val = min(np.min(all_targets), np.min(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Stiffness (GPa)')
    plt.ylabel('Predicted Stiffness (GPa)')
    plt.title('3D CNN Validation Performance')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.2)
    
    plt.savefig('validation_plot.png', dpi=300, bbox_inches='tight')
    print("-> Plot saved to validation_plot.png")

if __name__ == "__main__":
    plot_validation()
