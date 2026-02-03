import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.gatekeeper_model import GatekeeperModel
from src.models.final_maize_model import MaizeDiseaseModel

app = FastAPI(title="Maize Disease Diagnosis API", 
              description="A hierarchical ensemble for maize leaf pathology.")

# --- CONFIG & MODELS ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
GK_PATH = "checkpoints/gatekeeper/gatekeeper-epoch=11-gatekeeper_val_f1=0.95.ckpt"
HERO_PATH = "checkpoints/hero_model/maize-epoch=19-val_f1=0.66.ckpt"
LABEL_NAMES = ['GLS', 'NCLB', 'PLS', 'CR', 'SR', 'Healthy', 'Other', 'Unidentified']
THRESHOLDS = [0.3614, 0.4449, 0.2601, 0.2555, 0.9848, 0.6011, 0.2134, 0.2373]

# Load models into memory once at startup
print(f"📦 Loading models onto {DEVICE}...")
gk_model = GatekeeperModel.load_from_checkpoint(GK_PATH).to(DEVICE).eval()
hero_model = MaizeDiseaseModel.load_from_checkpoint(HERO_PATH).to(DEVICE).eval()

# Preprocessing pipeline
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read and preprocess image
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img_np = np.array(img)
    
    # Apply transforms
    input_tensor = transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # --- STAGE 1: Gatekeeper ---
        gk_prob = torch.sigmoid(gk_model(input_tensor)).item()
        
        # --- STAGE 2: Logic Branching ---
        if gk_prob < 0.3:
            return {
                "filename": file.filename,
                "status": "Healthy",
                "confidence": 1 - gk_prob,
                "diagnoses": [],
                "engine": "Gatekeeper only"
            }
        
        # Specialist Pass
        hero_probs = torch.sigmoid(hero_model(input_tensor)).squeeze().cpu().numpy()
        detected = []
        for i, p in enumerate(hero_probs):
            if p >= THRESHOLDS[i] and i != 5: # Exclude NoFoliar index
                detected.append({"label": LABEL_NAMES[i], "probability": float(p)})

        return {
            "filename": file.filename,
            "status": "Diseased" if detected else "Healthy (Low Symptom)",
            "gk_score": gk_prob,
            "diagnoses": detected,
            "engine": "Hierarchical Ensemble"
        }

@app.get("/health")
def health_check():
    return {"status": "online", "device": DEVICE}