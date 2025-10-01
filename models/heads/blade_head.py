# models/heads/blade_head.py (수정 완료)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SegFormerBladeHead(nn.Module):
    """Head-A: 블레이드 세그멘테이션 (다중 스케일 입력 지원)"""
    
    def __init__(self, in_channels: List[int], num_classes: int = 1, hidden_dim: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        
        # --- [수정] 다중 스케일 입력을 처리하도록 MLP 리스트 수정 ---
        self.mlps = nn.ModuleList()
        for ch in in_channels:
            mlp = nn.Sequential(
                nn.Conv2d(ch, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.mlps.append(mlp)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * len(in_channels), hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # Classifier
        self.classifier = nn.Conv2d(hidden_dim, num_classes, 1)
        
    def forward(self, features: List[torch.Tensor]):
        processed = []
        target_size = features[0].shape[-2:]
        
        for i, (feat, mlp) in enumerate(zip(features, self.mlps)):
            proc = mlp(feat)
            if proc.shape[-2:] != target_size:
                proc = F.interpolate(proc, size=target_size, mode='bilinear', align_corners=False)
            processed.append(proc)
        
        # Fusion
        fused = torch.cat(processed, dim=1)
        fused = self.fusion(fused)
        
        # Classification
        output = self.classifier(fused)
        
        # 최종 출력을 입력 이미지 크기(e.g., 640x640)에 맞게 업샘플링
        # 이 부분은 모델의 최종 출력 크기에 따라 달라질 수 있음
        # 여기서는 원본 forward 로직을 유지
        
        return output