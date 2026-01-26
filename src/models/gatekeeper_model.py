import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import torchmetrics

class GatekeeperModel(pl.LightningModule):
    def __init__(self, learning_rate=0.00047, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # Load backbone
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        
        # Freeze initial layers, fine-tune the top
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.features[6:].parameters():
            param.requires_grad = True
        
        # Binary Classification Head
        input_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=self.hparams.dropout),
            nn.Linear(input_features, 1) # Single output for Binary
        )
        
        # Binary Metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # We use BCEWithLogitsLoss for stable training
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        self.val_f1(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("gatekeeper_val_f1", self.val_f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.learning_rate
        )