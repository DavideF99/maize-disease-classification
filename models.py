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

class MaizeDiseaseModel(pl.LightningModule):
    def __init__(self, num_classes=8, learning_rate=0.00047, class_weights=None, 
                 label_smoothing=0.09, weight_decay=0.00011, dropout=0.5):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.learning_rate = learning_rate
        
        # EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(weights='DEFAULT')

        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for param in self.backbone.features[6:].parameters():
            param.requires_grad = True
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        
        input_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=self.hparams.dropout, inplace=True),
            nn.Linear(input_features, num_classes)
        )
        
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes)

        if class_weights is not None:
            self.register_buffer("weights", class_weights)
        else:
            self.register_buffer("weights", torch.ones(num_classes))

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        eps = self.hparams.label_smoothing
        y_smoothed = y * (1 - eps) + 0.5 * eps
        
        loss = F.binary_cross_entropy_with_logits(
            logits, y_smoothed, 
            pos_weight=self.weights,
            reduction='mean'
        )
        
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
