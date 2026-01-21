import pytorch_lightning as pl
import torch
import numpy as np
from src.data.maize_datamodule import MaizeDataModule
from src.models.maize_model import MaizeDiseaseModel
from sklearn.metrics import classification_report, precision_recall_curve
import os

def find_best_thresholds(all_probs, all_labels):
    """Finds the F1-optimizing threshold for each class individually."""
    best_thresholds = []
    # Loop through each of the 8 classes
    for i in range(all_labels.shape[1]):
        precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        
        # Calculate F1 score for every possible threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Get the index of the highest F1 score
        best_idx = np.argmax(f1_scores)
        
        # Guard against classes with zero labels in the set
        if best_idx < len(thresholds):
            best_thresholds.append(thresholds[best_idx])
        else:
            best_thresholds.append(0.5) # Default fallback
            
    return np.array(best_thresholds)

def get_predictions(model, dataloader):
    """Helper to collect all raw probabilities and labels from a loader."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            logits = model(x.to(model.device))
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    return np.vstack(all_probs), np.vstack(all_labels)

def evaluate():
    # 1. Load best model
    ckpt_dir = "checkpoints/"
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not checkpoints: return
    best_ckpt = os.path.join(ckpt_dir, sorted(checkpoints)[-1])
    model = MaizeDiseaseModel.load_from_checkpoint(best_ckpt)
    
    # 2. Setup Data
    dm = MaizeDataModule()
    dm.setup()
    
    # 3. PHASE 1: Find Best Thresholds on Validation Set
    print("Step 1: Optimizing thresholds on Validation Set...")
    val_probs, val_labels = get_predictions(model, dm.val_dataloader())
    best_thresholds = find_best_thresholds(val_probs, val_labels)
    
    # 4. PHASE 2: Test on Test Set using optimized thresholds
    print("Step 2: Evaluating on Test Set with optimized thresholds...")
    test_probs, test_labels = get_predictions(model, dm.test_dataloader())
    
    # Apply the thresholds (broadcasting across columns)
    test_preds = (test_probs > best_thresholds).astype(float)
    
    # 5. Report Results
    target_names = ['GLS', 'NCLB', 'PLS', 'CR', 'SR', 'NoFoliar', 'Other', 'Unidentified']
    print("\n--- Optimized Classification Report ---")
    print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))
    
    print("\n--- Discovered Optimal Thresholds ---")
    for name, thresh in zip(target_names, best_thresholds):
        print(f"{name}: {thresh:.4f}")

if __name__ == "__main__":
    evaluate()