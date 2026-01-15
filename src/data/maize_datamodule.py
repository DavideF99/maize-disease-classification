import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MaizeDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        # The 8 labels identified in our EDA
        self.label_columns = [
            'GLS', 'NCLB', 'PLS', 'CR', 'SR', 
            'NoFoliarSymptoms', 'Other', 'UnidentifiedDisease'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['filePath'])
        
        # Load image (OpenCV loads as BGR, we convert to RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract the 8 multi-label values as a float tensor
        labels = torch.tensor(row[self.label_columns].values.astype('float32'))
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, labels

class MaizeDataModule(pl.LightningDataModule):
    def __init__(self, csv_path='data/Database.csv', data_dir='data/raw', batch_size=32):
        super().__init__()
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Industry Standard: ImageNet normalization and 224x224 resize for MobileNet
        self.train_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.RandomResizedCrop(size=(224, 224)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.2), # Crucial for outdoor field images
            A.Rotate(limit=30, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        
        # Split into Train (80%) and Test (20%)
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        # Split Train further into Train (80% of 80%) and Val (20% of 80%)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
        
        if stage == "fit" or stage is None:
            self.train_set = MaizeDataset(train_df, self.data_dir, transform=self.train_transform)
            self.val_set = MaizeDataset(val_df, self.data_dir, transform=self.val_transform)
            
        if stage == "test" or stage is None:
            self.test_set = MaizeDataset(test_df, self.data_dir, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=2)