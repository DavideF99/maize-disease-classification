import matplotlib.pyplot as plt
import numpy as np
import torch
from src.data.maize_datamodule import MaizeDataModule

def denormalize(img):
    """Undo ImageNet normalization to visualize images correctly."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img.permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def run_check():
    # 1. Initialize DataModule
    # Ensure the paths match your project root
    dm = MaizeDataModule(csv_path='data/Database.csv', data_dir='data/raw', batch_size=4)
    dm.setup()
    
    # 2. Get a batch from the training loader
    loader = dm.train_dataloader()
    images, labels = next(iter(loader))
    
    # 3. Print shapes for validation
    print(f"--- DataModule Sanity Check ---")
    print(f"Image batch shape: {images.shape}")  # Expected: [4, 3, 224, 224]
    print(f"Label batch shape: {labels.shape}")  # Expected: [4, 8]
    print(f"Labels type: {labels.dtype}")        # Expected: torch.float32
    
    # 4. Map indices back to names for visualization
    class_names = [
        'GLS', 'NCLB', 'PLS', 'CR', 'SR', 
        'NoFoliarSymptoms', 'Other', 'UnidentifiedDisease'
    ]
    
    # 5. Visualize
    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(denormalize(images[i]))
        
        # Identify which diseases are present (where label == 1)
        active_labels = [class_names[j] for j, val in enumerate(labels[i]) if val == 1]
        title = "\n".join(active_labels) if active_labels else "No Labels"
        
        plt.title(f"Labels:\n{title}", fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('datamodule_check.png')
    print(f"--- Check Complete: Saved results to 'datamodule_check.png' ---")

if __name__ == "__main__":
    run_check()