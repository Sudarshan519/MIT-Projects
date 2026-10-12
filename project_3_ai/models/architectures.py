# ============================================================================
# INFO813 Project 3 — Model Architectures
# ============================================================================
# Defines two deep learning architectures for skin lesion classification:
#   1. CustomCNN: Baseline convolutional neural network built from scratch
#   2. EfficientNetB0Transfer: Pretrained transfer learning model with compound scaling
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights


# ============================================================================
# 1. Custom CNN (Baseline Model)
# ============================================================================
class CustomCNN(nn.Module):
    """
    4-Block Deep Convolutional Neural Network built from scratch.
    Designed specifically for 224x224 dermoscopic skin lesion images.
    
    Architecture:
      • Block 1: 2x [Conv(32, 3x3) + BatchNorm + ReLU] + MaxPool + Dropout(0.2)
      • Block 2: 2x [Conv(64, 3x3) + BatchNorm + ReLU] + MaxPool + Dropout(0.25)
      • Block 3: 2x [Conv(128, 3x3) + BatchNorm + ReLU] + MaxPool + Dropout(0.3)
      • Block 4: Conv(256, 3x3) + BatchNorm + ReLU + AdaptiveAvgPool + Dropout(0.4)
      • Classification Head: Linear(256, 128) + ReLU + Dropout(0.5) + Linear(128, num_classes)
    """
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.30)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.40)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits
        
    def get_last_conv_layer(self):
        """Returns the final convolutional layer for Grad-CAM."""
        return self.block4[0]


# ============================================================================
# 2. EfficientNet-B0 (Transfer Learning Model)
# ============================================================================
class EfficientNetB0Transfer(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet with custom medical classification head.
    Supports two-stage training:
      1. Feature extraction with frozen backbone
      2. Fine-tuning with unfrozen top convolutional layers
    """
    def __init__(self, num_classes=7, pretrained=True, dropout=0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        
        in_features = self.model.classifier[1].in_features  # 1280
        
        # Replace classifier with custom medical diagnosis head
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
        
    def freeze_backbone(self):
        """Freeze all feature extractor layers for Stage 1 training."""
        for param in self.model.features.parameters():
            param.requires_grad = False
            
    def unfreeze_top_layers(self, num_blocks=3):
        """
        Unfreeze the top `num_blocks` MBConv blocks for Stage 2 fine-tuning.
        EfficientNet-B0 features has 9 blocks (indices 0 to 8).
        """
        total_layers = len(self.model.features)
        cutoff = max(0, total_layers - num_blocks)
        for i, block in enumerate(self.model.features):
            for param in block.parameters():
                param.requires_grad = (i >= cutoff)
                
    def unfreeze_all(self):
        """Unfreeze all layers across the entire network."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_last_conv_layer(self):
        """Returns the final convolutional layer for Grad-CAM."""
        return self.model.features[-1]


def count_parameters(model):
    """Count total and trainable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
