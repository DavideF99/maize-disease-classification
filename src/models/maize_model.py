import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import torchmetrics

class MaizeDiseaseModel(pl.LightningModule):
    def __init__(self, num_classes=8, learning_rate=1e-4, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights']) # Ignore weights in hparams to avoid sizing issues
        self.learning_rate = learning_rate
        
        # 1. Load Pretrained MobileNetV3
        self.backbone = models.mobilenet_v3_large(weights='DEFAULT')

        # 2. Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze the last few blocks (features[13] to features[16] and the classifier)
        # In MobileNetV3, the later 'features' layers contain high-level semantic info
        for param in self.backbone.features[13:].parameters():
            param.requires_grad = True
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        
        # 3. Modify the classifier head for 8 classes
        input_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2), # 50% dropout is strong but effective for small datasets
            nn.Linear(input_features, num_classes)
        )
        
        # 4. Metrics
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes)

        # 5. Store weights for the loss function as a buffer
        # This ensures they move to MPS/GPU automatically
        if class_weights is not None:
            self.register_buffer("weights", class_weights)
        else:
            self.register_buffer("weights", torch.ones(num_classes))

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # We reference self.weights (the buffer) instead of class_weights
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.weights)
        
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
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5 # Adds L2 regularization
        )