import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader
from src.data.maize_datamodule import MaizeDataModule, MaizeDataset

class BinaryMaizeDataset(MaizeDataset):
    def __getitem__(self, idx):
        # 1. Get the original image and 8-class labels
        image, multi_labels = super().__getitem__(idx)
        
        # 2. Extract Healthy flag (Index 5)
        is_healthy = multi_labels[5] == 1.0
        
        # 3. Create Binary Label: 0.0 = Healthy, 1.0 = Diseased
        binary_label = torch.tensor([0.0 if is_healthy else 1.0], dtype=torch.float32)
        
        return image, binary_label

class BinaryMaizeDataModule(MaizeDataModule):
    def setup(self, stage=None):
        # Run original setup to get dataframes
        super().setup(stage)
        
        if stage == "fit" or stage is None:
            # Wrap datasets in binary logic
            self.train_set = BinaryMaizeDataset(self.train_set.df, self.data_dir, self.train_transform)
            self.val_set = BinaryMaizeDataset(self.val_set.df, self.data_dir, self.val_transform)

            # --- BALANCING LOGIC ---
            # Calculate weights for the training set ONLY
            train_df = self.train_set.df
            
            # Determine if each row is healthy or diseased
            # (Assuming the CSV column name is 'NoFoliarSymptoms')
            is_healthy_series = train_df['NoFoliarSymptoms'] == 1
            
            # Calculate counts
            num_healthy = is_healthy_series.sum()
            num_diseased = len(train_df) - num_healthy
            
            print(f"📊 Binary Stats - Healthy: {num_healthy}, Diseased: {num_diseased}")

            # Assign weights (Inverse of frequency)
            # We want: weight_healthy * num_healthy ≈ weight_diseased * num_diseased
            weight_healthy = 1.0 / (num_healthy + 1e-6)
            weight_diseased = 1.0 / (num_diseased + 1e-6)
            
            sample_weights = []
            for is_healthy in is_healthy_series:
                sample_weights.append(weight_healthy if is_healthy else weight_diseased)
            
            self.binary_sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            sampler=self.binary_sampler, # Use our new binary sampler
            num_workers=self.num_cpus,
            pin_memory=False,
            persistent_workers=True
        )