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

def get_predictions_with_tta(model, dataloader):
    """
    Helper to collect raw probabilities using Test-Time Augmentation (TTA).
    Averages original, horizontal flip, and vertical flip predictions.
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    device = model.device
    
    print(f"Running inference with TTA on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            
            # 1. Original Image pass
            probs1 = torch.sigmoid(model(x))
            
            # 2. Horizontal Flip pass (dim 3 is width)
            probs2 = torch.sigmoid(model(torch.flip(x, dims=[3])))
            
            # 3. Vertical Flip pass (dim 2 is height)
            probs3 = torch.sigmoid(model(torch.flip(x, dims=[2])))
            
            # Average the probabilities (3-way TTA)
            avg_probs = (probs1 + probs2 + probs3) / 3.0
            
            all_probs.append(avg_probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    return np.vstack(all_probs), np.vstack(all_labels)

def evaluate():
    # 1. Load best model from checkpoints
    ckpt_dir = "checkpoints/"
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory {ckpt_dir} not found.")
        return
        
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        print("No checkpoints found!")
        return
        
    # Picks the latest/best one based on filename
    best_ckpt = os.path.join(ckpt_dir, sorted(checkpoints)[-1])
    print(f"Loading best model from: {best_ckpt}")
    model = MaizeDiseaseModel.load_from_checkpoint(best_ckpt)
    
    # 2. Setup DataModule
    dm = MaizeDataModule()
    dm.setup()
    
    # 3. PHASE 1: Find Best Thresholds on Validation Set (Using TTA)
    # We use TTA during optimization because the probability distribution shifts when averaging
    print("\n--- Phase 1: Optimizing Thresholds (Validation Set + TTA) ---")
    val_probs, val_labels = get_predictions_with_tta(model, dm.val_dataloader())
    best_thresholds = find_best_thresholds(val_probs, val_labels)
    
    # 4. PHASE 2: Test on Test Set using optimized thresholds and TTA
    print("\n--- Phase 2: Final Evaluation (Test Set + TTA) ---")
    test_probs, test_labels = get_predictions_with_tta(model, dm.test_dataloader())
    
    # Apply optimized thresholds
    test_preds = (test_probs > best_thresholds).astype(float)
    
    # 5. Report Results
    target_names = ['GLS', 'NCLB', 'PLS', 'CR', 'SR', 'NoFoliar', 'Other', 'Unidentified']
    
    print("\n" + "="*40)
    print("MASTER EVALUATION REPORT (TTA + OPTIMIZED)")
    print("="*40)
    print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))
    
    print("\n--- Discovered Optimal Thresholds ---")
    for name, thresh in zip(target_names, best_thresholds):
        print(f"{name:<15}: {thresh:.4f}")

    from sklearn.metrics import multilabel_confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_multilabel_confusion_matrix(test_labels, test_preds, target_names):
        """Generates and saves a 2x2 confusion matrix for each class."""
        mcm = multilabel_confusion_matrix(test_labels, test_preds)
        
        # Create a grid for the 8 classes (4 rows, 2 columns)
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 18))
        axes = axes.flatten()
        
        for i, (matrix, name) in enumerate(zip(mcm, target_names)):
            sns.heatmap(matrix, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
            axes[i].set_title(f"Class: {name}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
            axes[i].set_xticklabels(['Absence', 'Presence'])
            axes[i].set_yticklabels(['Absence', 'Presence'])
        
        plt.tight_layout()
        plt.savefig("checkpoints/confusion_matrices.png")
        print("\nVisual confusion matrices saved to 'checkpoints/confusion_matrices.png'")

    plot_multilabel_confusion_matrix(test_labels, test_preds, target_names)
    
if __name__ == "__main__":
    evaluate()
    