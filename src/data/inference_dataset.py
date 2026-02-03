import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class MaizeInferenceDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        # Load your CSV
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        
        # Professional Transform Pipeline
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filePath'])
        
        image = cv2.imread(img_path)
        if image is None:
            # We must return the same data structure even for missing images
            # Use a dummy array of 8 zeros
            return torch.zeros((3, 224, 224)), row['filePath'], np.zeros(8, dtype=np.float32)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)["image"]
        
        # --- THE FIX: Explicitly cast to float32 ---
        actual_labels = row[['GLS','NCLB','PLS','CR','SR','NoFoliarSymptoms','Other','UnidentifiedDisease']].values.astype(np.float32)
        
        return transformed, row['filePath'], actual_labels