import pytorch_lightning as pl
import torch
from src.data.maize_datamodule import MaizeDataModule
from src.models.maize_model import MaizeDiseaseModel
import os

def evaluate():
    # 1. Find the best checkpoint file
    ckpt_dir = "checkpoints/"
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    # Picks the latest/best one
    best_ckpt = os.path.join(ckpt_dir, sorted(checkpoints)[-1])
    print(f"Loading best model from: {best_ckpt}")

    # 2. Load Model and Data
    # We load the weights into the model architecture
    model = MaizeDiseaseModel.load_from_checkpoint(best_ckpt)
    dm = MaizeDataModule()
    dm.setup(stage="test")

    # 3. Run Test
    trainer = pl.Trainer(accelerator="mps", devices=1)
    results = trainer.test(model, datamodule=dm)
    
    print("\n--- Final Test Results ---")
    print(results)

if __name__ == "__main__":
    evaluate()