import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.inference_dataset import MaizeInferenceDataset
from src.models.gatekeeper_model import GatekeeperModel
from src.models.final_maize_model import MaizeDiseaseModel

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# CONFIG
CSV_PATH = "data/Database.csv" 
IMG_DIR = "data/raw"
OUTPUT_FILE = "outputs/batch_verification_report.csv"
BATCH_SIZE = 32
DEVICE = "mps"
GK_THRESH = 0.3
THRESHOLDS = [0.3614, 0.4449, 0.2601, 0.2555, 0.9848, 0.6011, 0.2134, 0.2373]
LABEL_NAMES = ['GLS', 'NCLB', 'PLS', 'CR', 'SR', 'Healthy', 'Other', 'Unidentified']

def run_batch_verification():
    # 1. Models
    gk = GatekeeperModel.load_from_checkpoint("checkpoints/gatekeeper/gatekeeper-epoch=11-gatekeeper_val_f1=0.95.ckpt").to(DEVICE).eval()
    hero = MaizeDiseaseModel.load_from_checkpoint("checkpoints/hero_model/maize-epoch=19-val_f1=0.66.ckpt").to(DEVICE).eval()

    # 2. Data
    dataset = MaizeInferenceDataset(CSV_PATH, IMG_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

    results = []

    print(f"🔬 Verifying {len(dataset)} images via Batch Pipeline...")

    with torch.no_grad():
        for images, paths, actuals in tqdm(loader):
            images = images.to(DEVICE)
            
            # STAGE 1 & 2
            gk_probs = torch.sigmoid(gk(images)).squeeze()
            # Handle edge case for batch size 1
            if gk_probs.ndim == 0: gk_probs = gk_probs.unsqueeze(0)
            
            hero_probs = torch.sigmoid(hero(images))
            
            for i in range(len(paths)):
                # If image was missing (all zeros), skip
                if torch.all(images[i] == 0): continue
                
                # Use the correct variable name: GK_THRESH
                if gk_probs[i] < GK_THRESH:
                    pred_status = "Healthy"
                    pred_diseases = []
                else:
                    pred_status = "Diseased"
                    p = hero_probs[i].cpu().numpy()
                    pred_diseases = [LABEL_NAMES[j] for j in range(8) if p[j] >= THRESHOLDS[j] and j != 5]

                # labels are now tensors/numpy arrays from the loader
                actual_row = actuals[i]
                # Index 5 is NoFoliarSymptoms
                actual_status = "Healthy" if actual_row[5] == 1 else "Diseased"
                
                actual_disease_names = [LABEL_NAMES[j] for j in range(8) if actual_row[j] == 1 and j != 5]

                results.append({
                    "File": paths[i],
                    "Actual_Status": actual_status,
                    "Pred_Status": pred_status,
                    "Actual_Diseases": ", ".join(actual_disease_names) if actual_disease_names else "None",
                    "Pred_Diseases": ", ".join(pred_diseases) if pred_diseases else "None",
                    "Match": (actual_status == pred_status)
                })

    # 3. Save Structured Report
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Report generated: {OUTPUT_FILE}")

    # 4. Analyze results
    analyze_results(df)

def analyze_results(df):
    print("\n--- 📊 Statistical Analysis ---")
    
    # 1. Overall Binary Accuracy (Gatekeeper Performance)
    acc = accuracy_score(df['Actual_Status'], df['Pred_Status'])
    print(f"Gatekeeper Binary Accuracy: {acc:.2%}")

    # 2. Detailed Classification Report
    # We compare the 'Actual_Diseases' string vs 'Pred_Diseases'
    print("\nDetailed Performance Report:")
    print(classification_report(df['Actual_Status'], df['Pred_Status']))

    # 3. Create a Confusion Matrix for Binary Status
    cm = confusion_matrix(df['Actual_Status'], df['Pred_Status'], labels=["Healthy", "Diseased"])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Healthy", "Diseased"], 
                yticklabels=["Healthy", "Diseased"])
    plt.title("Gatekeeper Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    plot_path = "outputs/confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"\n📈 Confusion Matrix saved to: {plot_path}")

if __name__ == "__main__":
    run_batch_verification()