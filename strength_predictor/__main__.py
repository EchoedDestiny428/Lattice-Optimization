import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import GLU3DLatticeDataset
from .models import LatticeStrengthCNN3D

def main():
    parser = argparse.ArgumentParser(description="Material-Agnostic Engine CLI")
    parser.add_argument("action", choices=["train"])
    parser.add_argument("train_file")
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    # Load datasets
    train_dataset = GLU3DLatticeDataset(args.train_file, normalizer=None, augment=True)
    val_dataset = GLU3DLatticeDataset(args.val_file, normalizer=train_dataset.normalizer, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = LatticeStrengthCNN3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0

    print("\nStarting Training Loops...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for voxels, vfrac, labels in train_loader:
            voxels, vfrac, labels = voxels.to(device), vfrac.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(voxels, vfrac)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * voxels.size(0)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for voxels, vfrac, labels in val_loader:
                voxels, vfrac, labels = voxels.to(device), vfrac.to(device), labels.to(device)
                outputs = model(voxels, vfrac)
                val_loss += criterion(outputs, labels).item() * voxels.size(0)

        epoch_train = train_loss / len(train_loader.dataset)
        epoch_val = val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train MSE: {epoch_train:.6f} | Val MSE: {epoch_val:.6f}")

        if epoch_val < best_loss:
            best_loss = epoch_val
            patience_counter = 0
            torch.save(model.state_dict(), "best_strength_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] Triggered after {args.patience} epochs without validation progress.")
                break

    print("\nDone! Best model saved to: best_strength_model.pt")

if __name__ == "__main__":
    main()