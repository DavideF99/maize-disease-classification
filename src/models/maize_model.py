import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import torchmetrics

class MaizeDiseaseModel(pl.LightningModule):
    def __init__(self, num_classes=8, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # 1. Load Pretrained MobileNetV3
        # Using 'weights' is the modern PyTorch way (2026 standard)
        self.backbone = models.mobilenet_v3_large(weights='DEFAULT')
        
        # 2. Modify the classifier head for 8 classes
        # MobileNetV3 classifier is the last part of the network
        input_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(input_features, num_classes)
        
        # 3. Metrics (Industry Standard for multi-label)
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # Use Binary Cross Entropy with Logits for multi-label
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        self.train_f1(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        self.val_f1(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # We reuse validation logic but change the log labels to "test"
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Use a separate metric or just log with "test_" prefix
        self.val_f1(logits, y) # It's okay to reuse the metric object here
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_f1", self.val_f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)