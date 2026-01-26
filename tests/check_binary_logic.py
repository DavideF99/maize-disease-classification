from src.data.binary_maize_datamodule import BinaryMaizeDataModule
import torch

def test_logic():
    dm = BinaryMaizeDataModule(batch_size=4)
    dm.setup()
    
    loader = dm.train_dataloader()
    images, labels = next(iter(loader))
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels from batch: \n{labels}")
    
    # Check if we are getting both 0s and 1s thanks to the sampler
    unique_labels = torch.unique(labels)
    print(f"Unique labels in this batch: {unique_labels.tolist()}")
    
    if len(unique_labels) > 1:
        print("✅ Success: Sampler is providing a mix of Healthy and Diseased!")
    else:
        print("⚠️ Warning: Batch only contains one class. Check sampler weights if this repeats.")

if __name__ == "__main__":
    test_logic()