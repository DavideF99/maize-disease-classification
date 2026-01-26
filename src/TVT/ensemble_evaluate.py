import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_curve
from src.models.gatekeeper_model import GatekeeperModel
from src.models.final_maize_model import MaizeDiseaseModel
from src.data.maize_datamodule import MaizeDataModule

# --- CONFIG ---
GATEKEEPER_PATH = "checkpoints/gatekeeper/gatekeeper-epoch=11-gatekeeper_val_f1=0.95.ckpt"
HERO_PATH = "checkpoints/hero_model/maize-epoch=19-val_f1=0.66.ckpt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
GK_THRESHOLD = 0.3 # Safety margin: if prob > 0.3, send to specialist

def get_ensemble_probs_with_tta(gk_model, hero_model, dataloader):
    gk_model.eval()
    hero_model.eval()
    
    all_final_probs = []
    all_labels = []
    
    target_names = ['GLS', 'NCLB', 'PLS', 'CR', 'SR', 'NoFoliar', 'Other', 'Unidentified']

    print(f"Running Ensemble + TTA Inference on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            x = x.to(DEVICE)
            
            # --- 1. TTA PASSES ---
            # Original, Horizontal Flip, Vertical Flip
            inputs = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]
            
            batch_gk_probs = []
            batch_hero_probs = []
            
            for aug_x in inputs:
                # Gatekeeper Probabilities
                gk_p = torch.sigmoid(gk_model(aug_x))
                batch_gk_probs.append(gk_p)
                
                # Hero Specialist Probabilities
                hero_p = torch.sigmoid(hero_model(aug_x))
                batch_hero_probs.append(hero_p)
            
            # Average TTA results
            avg_gk_p = torch.stack(batch_gk_probs).mean(dim=0).squeeze().cpu().numpy()
            avg_hero_p = torch.stack(batch_hero_probs).mean(dim=0).cpu().numpy()
            
            # Ensure avg_gk_p is iterable for single-item batches
            if avg_gk_p.ndim == 0: avg_gk_p = np.array([avg_gk_p])

            # --- 2. ENSEMBLE LOGIC ---
            final_probs = np.zeros_like(avg_hero_p)
            
            for i in range(len(avg_gk_p)):
                if avg_gk_p[i] < GK_THRESHOLD:
                    # Gatekeeper says Healthy: Set NoFoliar (idx 5) to 1.0, others 0.0
                    final_probs[i, 5] = 1.0
                else:
                    # Gatekeeper says Diseased: Use Hero Model Probs
                    final_probs[i] = avg_hero_p[i]
            
            all_final_probs.append(final_probs)
            all_labels.append(y.cpu().numpy())
            
    return np.vstack(all_final_probs), np.vstack(all_labels)

def find_best_thresholds(all_probs, all_labels):
    best_thresholds = []
    for i in range(all_labels.shape[1]):
        # NoFoliar (idx 5) is handled by Gatekeeper, but we optimize it anyway
        precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresholds.append(thresholds[best_idx] if best_idx < len(thresholds) else 0.5)
    return np.array(best_thresholds)

def run_master_evaluation():
    # 1. Load Models
    gk = GatekeeperModel.load_from_checkpoint(GATEKEEPER_PATH).to(DEVICE)
    hero = MaizeDiseaseModel.load_from_checkpoint(HERO_PATH).to(DEVICE)
    
    # 2. Setup Data
    dm = MaizeDataModule()
    dm.setup()
    
    # 3. Phase 1: Dynamic Threshold Discovery on Validation Set
    print("\n--- Phase 1: Dynamic Thresholding (Validation + TTA) ---")
    val_probs, val_labels = get_ensemble_probs_with_tta(gk, hero, dm.val_dataloader())
    best_thresholds = find_best_thresholds(val_probs, val_labels)
    
    # 4. Phase 2: Final Evaluation on Test Set
    print("\n--- Phase 2: Final Ensemble Evaluation (Test + TTA) ---")
    test_probs, test_labels = get_ensemble_probs_with_tta(gk, hero, dm.test_dataloader())
    
    # Apply dynamic thresholds
    test_preds = (test_probs >= best_thresholds).astype(float)
    
    # 5. Report
    target_names = ['GLS', 'NCLB', 'PLS', 'CR', 'SR', 'NoFoliar', 'Other', 'Unidentified']
    print("\n" + "="*50)
    print("MASTER ENSEMBLE REPORT (GATEKEEPER + HERO + TTA + DYNAMIC)")
    print("="*50)
    print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))
    
    print("\n--- New Dynamic Thresholds ---")
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
        plt.savefig("checkpoints/gatekeeper/confusion_matrices.png")
        print("\nVisual confusion matrices saved to 'checkpoints/gatekeeper/confusion_matrices.png'")

    plot_multilabel_confusion_matrix(test_labels, test_preds, target_names)

if __name__ == "__main__":
    run_master_evaluation()