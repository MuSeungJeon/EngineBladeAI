# models/heads/blade_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerBladeHead(nn.Module):
    """Head-A: 블레이드 세그멘테이션"""
    
    def __init__(self, in_channels=256, num_classes=2, dropout_rate=0.1):
        super().__init__()
        
        hidden_dim = 256
        
        # Multi-scale MLPs
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # Classifier
        self.classifier = nn.Conv2d(hidden_dim, num_classes, 1)
        
    def forward(self, features):
        # Process each scale
        processed = []
        target_size = features[0].shape[-2:]
        
        for feat, mlp in zip(features, self.mlps):
            proc = mlp(feat)
            if proc.shape[-2:] != target_size:
                proc = F.interpolate(proc, size=target_size, mode='bilinear', align_corners=False)
            processed.append(proc)
        
        # Fusion
        fused = torch.cat(processed, dim=1)
        fused = self.fusion(fused)
        
        # Classification
        output = self.classifier(fused)
        output = F.interpolate(output, size=(640, 640), mode='bilinear', align_corners=False)
        
        return output