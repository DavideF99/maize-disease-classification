import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from src.models.maize_model import MaizeDiseaseModel

def get_best_checkpoint(checkpoint_dir="checkpoints/"):
    """Dynamically finds the best/latest checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        return None
    # Sorts by name - usually Lightning puts the best F1 or latest epoch at the end
    return os.path.join(checkpoint_dir, sorted(checkpoints)[-1])

def predict(image_path, checkpoint_path):
    # 1. Setup metadata & Thresholds from your Master Evaluation
    class_names = ["GLS", "NCLB", "PLS", "CR", "SR", "NoFoliar", "Other", "Unidentified"]
    
    # These are the specific 'optimal' thresholds you discovered:
    thresholds = np.array([0.1696, 0.3582, 0.2581, 0.2542, 0.8329, 0.7460, 0.2508, 0.1324])
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Load model
    print(f"📦 Loading model from: {checkpoint_path}")
    model = MaizeDiseaseModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # 3. Prepare Image (Using 448 to match training)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(height=448, width=448),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # 4. TTA Inference
    # We perform three passes and average the probabilities
    with torch.no_grad():
        # Pass 1: Original
        input_orig = transform(image=image)['image'].unsqueeze(0).to(device)
        prob1 = torch.sigmoid(model(input_orig))

        # Pass 2: Horizontal Flip
        input_hflip = torch.flip(input_orig, dims=[3])
        prob2 = torch.sigmoid(model(input_hflip))

        # Pass 3: Vertical Flip
        input_vflip = torch.flip(input_orig, dims=[2])
        prob3 = torch.sigmoid(model(input_vflip))

        # Average results
        avg_probs = ((prob1 + prob2 + prob3) / 3.0).squeeze().cpu().numpy()

    # 5. Show Results
    print(f"\nFinal Diagnosis for: {os.path.basename(image_path)}")
    print("=" * 40)
    
    found_any = False
    for name, prob, thresh in zip(class_names, avg_probs, thresholds):
        status = " [POS]" if prob >= thresh else "      "
        if prob >= thresh or prob > 0.10: # Show positive hits or anything > 10%
            print(f"{status} {name:<12}: {prob*100:>5.1f}% (Threshold: {thresh*100:.1f}%)")
            found_any = True
            
    if not found_any:
        print("No diseases detected with high confidence.")
    print("=" * 40)

if __name__ == "__main__":
    import argparse
    
    # Use argparse to allow passing path via terminal: python src/predict.py --path my_leaf.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the image file")
    args = parser.parse_args()

    # Dynamic Path Handling
    ckpt = get_best_checkpoint()
    
    if not ckpt:
        print("❌ Error: No checkpoint found in 'checkpoints/'")
    else:
        # If user didn't provide a path, look for the first image in data/raw
        test_img = args.path
        if not test_img:
            raw_dir = "data/raw"
            images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                test_img = os.path.join(raw_dir, images[0])
            else:
                print(f"❌ Error: No image provided and none found in {raw_dir}")
                exit()

        print(f"🔍 Analyzing: {test_img}")
        try:
            predict(test_img, ckpt)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")