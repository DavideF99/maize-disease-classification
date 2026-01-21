import pytorch_lightning as pl
import torch
import numpy as np
from src.data.maize_datamodule import MaizeDataModule
from src.models.maize_model import MaizeDiseaseModel
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import os

def evaluate():
    ckpt_dir = "checkpoints/"
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    best_ckpt = os.path.join(ckpt_dir, sorted(checkpoints)[-1])
    print(f"Loading best model from: {best_ckpt}")

    # Load Model and Data
    model = MaizeDiseaseModel.load_from_checkpoint(best_ckpt)
    model.eval() # Set to evaluation mode
    
    dm = MaizeDataModule()
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    all_preds = []
    all_labels = []

    print("Running detailed inference...")
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            # Move to device (MPS for Mac)
            x = x.to(model.device)
            logits = model(x)

            # Define thresholds for each class (Must match the order of your classes)
            # Order: GLS, NCLB, PLS, CR, SR, NoFoliar, Other, Unidentified
            thresholds = torch.tensor([0.35, 0.35, 0.35, 0.35, 0.20, 0.70, 0.30, 0.15]).to(model.device)
            
            # Convert logits to probabilities
            probs = torch.sigmoid(logits)
            
            # Apply thresholds class-by-class
            # This compares each column of 'probs' to the corresponding value in 'thresholds'
            preds = (probs > thresholds).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # Concatenate results
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    # Class names from your DataModule
    target_names = ['GLS', 'NCLB', 'PLS', 'CR', 'SR', 'NoFoliar', 'Other', 'Unidentified']

    print("\n--- Per-Class Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    evaluate()