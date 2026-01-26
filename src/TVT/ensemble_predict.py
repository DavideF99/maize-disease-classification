import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.gatekeeper_model import GatekeeperModel
from src.models.final_maize_model import MaizeDiseaseModel

# --- FINAL PRODUCTION CONFIG ---
GATEKEEPER_PATH = "checkpoints/gatekeeper/gatekeeper-epoch=11-gatekeeper_val_f1=0.95.ckpt"
HERO_PATH = "checkpoints/hero_model/maize-epoch=19-val_f1=0.66.ckpt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

LABEL_NAMES = ['GLS', 'NCLB', 'PLS', 'CR', 'SR', 'Healthy', 'Other', 'Unidentified']
THRESHOLDS = [0.3614, 0.4449, 0.2601, 0.2555, 0.9848, 0.6011, 0.2134, 0.2373]
GK_SAFE_THRESH = 0.3

class MaizePredictor:
    def __init__(self):
        print(f"📦 Loading Ensemble Models on {DEVICE}...")
        self.gk = GatekeeperModel.load_from_checkpoint(GATEKEEPER_PATH).to(DEVICE).eval()
        self.hero = MaizeDiseaseModel.load_from_checkpoint(HERO_PATH).to(DEVICE).eval()
        
        # Standard Preprocessing
        self.base_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _get_tta_variants(self, image):
        """Generates 3 views of the image for robust prediction."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (224, 224))
        
        v1 = self.base_transform(image=resized)["image"]
        v2 = torch.flip(v1, dims=[2]) # Horizontal
        v3 = torch.flip(v1, dims=[1]) # Vertical
        
        return torch.stack([v1, v2, v3]).to(DEVICE)

    def predict(self, img_path):
        image = cv2.imread(str(img_path))
        if image is None: return "Error: Image not found."

        with torch.no_grad():
            batch = self._get_tta_variants(image)
            
            # 1. Gatekeeper TTA Pass
            gk_probs = torch.sigmoid(self.gk(batch)).mean().item()
            
            if gk_probs < GK_SAFE_THRESH:
                return {"status": "Healthy", "diseases": [], "confidence": 1 - gk_probs}

            # 2. Hero Specialist TTA Pass
            hero_probs = torch.sigmoid(self.hero(batch)).mean(dim=0).cpu().numpy()
            
            detected = []
            for i, prob in enumerate(hero_probs):
                if prob >= THRESHOLDS[i] and i != 5: # Skip NoFoliar index in disease list
                    detected.append(f"{LABEL_NAMES[i]} ({prob*100:.1f}%)")

            if not detected:
                return {"status": "Healthy (Low Symptom)", "diseases": [], "confidence": 1 - gk_probs}
            
            return {"status": "Diseased", "diseases": detected, "gk_score": gk_probs}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to maize leaf image")
    args = parser.parse_args()

    predictor = MaizePredictor()
    result = predictor.predict(args.image)

    print("\n--- 🌽 Maize Disease Diagnosis ---")
    print(f"Status: {result['status']}")
    if result['diseases']:
        print("Detected Symptoms:")
        for d in result['diseases']:
            print(f" - {d}")
    print("----------------------------------\n")

if __name__ == "__main__":
    main()