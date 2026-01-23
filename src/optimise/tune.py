import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from src.models.maize_model import MaizeDiseaseModel
from src.data.maize_datamodule import MaizeDataModule

def objective(trial):
    # 1. Define the search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # 2. Initialize Data and Model with suggested hparams
    # Note: You'll need to update your MaizeDiseaseModel to accept these in __init__
    dm = MaizeDataModule(batch_size=batch_size)
    model = MaizeDiseaseModel(
        learning_rate=learning_rate,  # Match the model's argument name
        label_smoothing=smoothing, 
        weight_decay=weight_decay,
        dropout=dropout
    )

    # 3. Setup Trainer with Pruning
    # The PruningCallback stops the trial if the results are worse than previous trials
    trainer = pl.Trainer(
        max_epochs=7,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        # IMPORTANT: val_f1 is only available AFTER the first validation epoch
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_f1"),
            EarlyStopping(monitor="val_f1", mode="max", patience=3)
        ]
    )

    # 4. Train
    trainer.fit(model, datamodule=dm)

    # 5. Return the metric we want to MAXIMIZE, Use .get() to avoid KeyError if a trial fails early
    return trainer.callback_metrics.get("val_f1", 0.0).item()

if __name__ == "__main__":
    # Create or load a persistent study that maximize the F1 score
    study = optuna.create_study(
        study_name="maize_optimization",
        storage="sqlite:///maize_opt.db", # This creates a file in your folder
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    print("🚀 Starting Optimization...")
    study.optimize(objective, n_trials=30)

    print("\n" + "="*30)
    print("BEST TRIAL RESULTS")
    print(f"Best F1 Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")