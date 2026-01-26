import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.gatekeeper_model import GatekeeperModel
from src.data.binary_maize_datamodule import BinaryMaizeDataModule

def train_gatekeeper():
    # We can reuse your optimized Hero hparams
    datamodule = BinaryMaizeDataModule(batch_size=16)
    model = GatekeeperModel(learning_rate=0.00047, dropout=0.5)

    checkpoint_callback = ModelCheckpoint(
        monitor="gatekeeper_val_f1",
        dirpath="checkpoints/gatekeeper/",
        filename="gatekeeper-{epoch:02d}-{gatekeeper_val_f1:.2f}",
        save_top_k=1,
        mode="max",
    )

    logger = TensorBoardLogger("logs", name="gatekeeper_run")
    
    trainer = pl.Trainer(
        max_epochs=40,
        accelerator="mps",
        devices=1,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="gatekeeper_val_f1", patience=7, mode="max")],
        logger=logger,
        precision="16-mixed"
    )

    print("🛡️ Training Gatekeeper (Healthy vs. Diseased)...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train_gatekeeper()