import os
import torch
import pandas as pd
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import cv2

class MaizeDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.label_columns = [
            'GLS', 'NCLB', 'PLS', 'CR', 'SR', 
            'NoFoliarSymptoms', 'Other', 'UnidentifiedDisease'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Optimization: use .iloc sparingly or convert df to list of dicts for speed
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['filePath'])
        
        # cv2.imread is faster than PIL for Albumentations
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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
        self.num_cpus = os.cpu_count()

        # Augmentation Strategy: CROP FIRST for speed
        self.train_transform = A.Compose([
            # 1. Immediate Reduction (Crucial for Speed)
            A.RandomResizedCrop(size=(448, 448), scale=(0.4, 1.0), p=1.0),
            
            # 2. Geometric (Standard for Leaves)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            
            # 3. Visual Noise (Prevents overfitting to light/shadow)
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Resize(height=448, width=448),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
        
        if stage == "fit" or stage is None:
            self.train_set = MaizeDataset(train_df, self.data_dir, transform=self.train_transform)
            self.val_set = MaizeDataset(val_df, self.data_dir, transform=self.val_transform)

            # Sampling Strategy for Southern Rust (SR) and NoFoliar
            weights = []
            for _, row in train_df.iterrows():
                if row['SR'] == 1 or row['NoFoliarSymptoms'] == 1:
                    weights.append(10.0)
                else:
                    weights.append(1.0)
            
            self.sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(weights),
                num_samples=len(weights),
                replacement=True
            )
            
        if stage == "test" or stage is None:
            self.test_set = MaizeDataset(test_df, self.data_dir, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            sampler=self.sampler,
            num_workers=self.num_cpus,
            pin_memory=False,            # Not needed for MPS due to CPU and GPU sharing memory
            persistent_workers=True      # Essential for short Optuna trials
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_cpus,
            pin_memory=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_cpus,
            pin_memory=False,
            persistent_workers=True
        )