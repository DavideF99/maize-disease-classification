import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.maize_datamodule import MaizeDataModule
from src.models.maize_model import MaizeDiseaseModel

def train():
    # 1. Initialize Data and Model
    dm = MaizeDataModule(batch_size=32)
    model = MaizeDiseaseModel(learning_rate=1e-3)

    # 2. Setup Loggers and Callbacks
    logger = TensorBoardLogger("logs", name="maize_disease_v1")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-maize-model-{val_f1:.2f}",
        save_top_k=1,
        mode="max",
        monitor="val_f1"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        patience=5,
        mode="max"
    )

    # 3. Initialize Trainer
    # We use 'mps' for your MacBook GPU and 'devices=1'
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="mps",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        precision="16-mixed", # Uses mixed precision to speed up M-series chips
        log_every_n_steps=5  # Add this for smoother real-time charts on TensorBoard
    )

    # 4. Start Training
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    train()