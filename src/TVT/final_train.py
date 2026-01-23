import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.final_maize_model import MaizeDiseaseModel
from src.data.maize_datamodule import MaizeDataModule

def train():
    # 1. Best Hyperparameters from Optuna
    hyperparams = {
        "learning_rate": 0.00047,
        "label_smoothing": 0.09,
        "weight_decay": 0.00011,
        "dropout": 0.5, 
    }

    batch_size = 16

    # 2. Setup Data and Model
    datamodule = MaizeDataModule(batch_size=batch_size)
    model = MaizeDiseaseModel(**hyperparams)

    # 3. Intelligent Callbacks
    # Save the best model based on val_f1 (not just the last epoch)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        dirpath="checkpoints/hero_model/",
        filename="maize-{epoch:02d}-{val_f1:.2f}",
        save_top_k=1,
        mode="max",
    )

    # Stop if the model doesn't improve for 10 validation checks
    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        patience=10, 
        mode="max",
        verbose=True
    )

    # 4. Initialize Trainer
    logger = TensorBoardLogger("logs", name="hero_run")
    
    trainer = pl.Trainer(
        max_epochs=100,              # We set high and let EarlyStopping handle it
        accelerator="mps",           # Your Mac GPU
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, LearningRateMonitor(logging_interval='epoch')],
        logger=logger,
        precision="16-mixed",        # Speeds up training on Apple Silicon
        log_every_n_steps=10
    )

    # 5. Launch Training
    print("🚀 Starting Hero Training with Optimized Hyperparameters...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train()