import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import torchmetrics

class MaizeDiseaseModel(pl.LightningModule):
    def __init__(self, num_classes=8, learning_rate=1e-4, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights']) # Ignore weights in hparams to avoid sizing issues during checkpoint loading
        self.learning_rate = learning_rate
        
        # 1. Load Pretrained EfficientNet-B0
        # EfficientNet-B0 is more robust than MobileNet, using better compound scaling
        self.backbone = models.efficientnet_b0(weights='DEFAULT')

        # 2. Freeze base layers for stable initial transfer learning
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze the final feature blocks and the classifier
        # This allows the model to fine-tune high-level disease features 
        # while keeping general image features (edges/shapes) frozen.
        for param in self.backbone.features[6:].parameters():
            param.requires_grad = True
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        
        # 3. Modify the classifier head for 8 classes
        # EfficientNet-B0's classifier is a Sequential: [Dropout, Linear]
        # We replace the final Linear layer (index 1) with our task-specific head
        input_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), # Higher dropout to prevent overfitting on rare classes
            nn.Linear(input_features, num_classes)
        )
        
        # 4. Multi-label Metrics
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes)

        # 5. Store weights for the loss function as a buffer to handle device movement
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
        # Use weighted loss to continue focusing on minority classes like SR
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
        # Filter for only trainable parameters to save memory
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # 6. Learning Rate Scheduler (ReduceLROnPlateau)
        # This monitors 'val_loss' and cuts the LR by half if it plateaus for 3 epochs.
        # This helps the model "settle" into the best weights at the end of training.
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
                "monitor": "val_loss", # Required for ReduceLROnPlateau in Lightning
            },
        }