import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from src.models.maize_model import MaizeDiseaseModel

def predict(image_path, checkpoint_path):
    # 1. Setup metadata
    class_names = ["GLS", "NCLB", "PLS", "CR", "SR", "NoFoliar", "Other", "Unidentified"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Load model
    model = MaizeDiseaseModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # 3. Prepare Image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    input_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # 5. Show Results
    print(f"\nResults for: {image_path}")
    print("-" * 30)
    for name, prob in zip(class_names, probs):
        if prob > 0.01: # Showing anything with over 1% confidence
            print(f"{name}: {prob*100:.2f}% (confidence)")

if __name__ == "__main__":
    import os
    
    # 1. Setup paths
    raw_data_path = "data/raw"
    checkpoint = "checkpoints/best-maize-model-val_f1=0.56.ckpt"
    
    # 2. Find the first available image in the folder
    available_images = [f for f in os.listdir(raw_data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not available_images:
        print(f"❌ Error: No images found in {raw_data_path}. Please check your folder structure.")
    else:
        test_image = os.path.join(raw_data_path, available_images[0])
        print(f"🔍 Testing with found image: {test_image}")
        
        # 3. Run prediction
        try:
            predict(image_path=test_image, checkpoint_path=checkpoint)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")