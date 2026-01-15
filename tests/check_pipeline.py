import pytorch_lightning as pl
from src.data.maize_datamodule import MaizeDataModule
from src.models.maize_model import MaizeDiseaseModel

def run_pipeline_check():
    print("--- Pipeline Smoke Test ---")
    dm = MaizeDataModule(batch_size=4)
    model = MaizeDiseaseModel()

    # 'fast_dev_run=True' is the professional way to test the full loop
    trainer = pl.Trainer(
        fast_dev_run=True, 
        accelerator="mps", 
        devices=1
    )
    
    try:
        trainer.fit(model, datamodule=dm)
        print("✅ Success: The full training loop is functional.")
    except Exception as e:
        print(f"❌ Failure: Pipeline crashed. Error: {e}")

if __name__ == "__main__":
    run_pipeline_check()