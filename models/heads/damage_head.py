# models/heads/damage_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BladeGuidedAttention(nn.Module):
    """블레이드 가이드 어텐션 메커니즘"""
    
    def __init__(self, channels=256):
        super().__init__()
        
        # Spatial attention with blade mask
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 3, padding=1),  # +1 for blade mask
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, blade_mask):
        # Ensure blade_mask has correct dimensions
        if blade_mask.dim() == 3:
            blade_mask = blade_mask.unsqueeze(1)
        
        # Spatial attention using blade mask
        concat = torch.cat([features, blade_mask], dim=1)
        spatial_att = self.spatial_attention(concat)
        
        # Preserve boundary regions (dilated mask)
        dilated_mask = F.max_pool2d(blade_mask, kernel_size=3, stride=1, padding=1)
        spatial_att = torch.maximum(spatial_att, dilated_mask * 0.5)
        
        # Channel attention
        channel_att = self.channel_attention(features)
        
        # Apply both attentions
        refined = features * spatial_att * channel_att
        
        return refined, spatial_att


class DamageDetectionHead(nn.Module):
    """Head-B: 손상 검출 (멀티라벨 + 세그멘테이션)"""
    
    def __init__(
        self, 
        in_channels=256, 
        num_classes=3,
        use_soft_gating=True,
        boundary_margin=10
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_soft_gating = use_soft_gating
        self.boundary_margin = boundary_margin
        
        # Blade-guided attention module
        if use_soft_gating:
            self.blade_attention = BladeGuidedAttention(in_channels * 4)
            
            # Learnable gate weight for soft gating
            self.gate_weight = nn.Parameter(torch.tensor(0.7))
            
            # Soft mask refinement
            self.mask_refiner = nn.Sequential(
                nn.Conv2d(1, 16, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 5, padding=2),
                nn.Sigmoid()
            )
        
        # Feature fusion for segmentation
        self.seg_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Boundary enhancement module
        self.boundary_enhancer = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes + 1, 1)  # +1 for background
        )
        
        # Multi-label classification head
        self.multilabel_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def expand_mask(self, mask, margin):
        """마스크를 margin 픽셀만큼 확장"""
        if margin <= 0:
            return mask
            
        kernel_size = 2 * margin + 1
        expanded = F.max_pool2d(
            mask, 
            kernel_size=kernel_size,
            stride=1,
            padding=margin
        )
        return expanded
    
    def create_soft_mask(self, blade_mask):
        """소프트 마스크 생성 (경계 부드럽게)"""
        # 1. Expand mask with margin
        expanded = self.expand_mask(blade_mask, self.boundary_margin)
        
        # 2. Apply Gaussian blur for smooth boundaries
        # Create Gaussian kernel
        kernel_size = 7
        sigma = 2.0
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        gaussian_kernel = (g.view(1, -1) * g.view(-1, 1)).view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.to(blade_mask.device)
        
        # Apply Gaussian blur
        soft_mask = F.conv2d(expanded, gaussian_kernel, padding=kernel_size//2)
        
        # 3. Refine with learnable module
        if hasattr(self, 'mask_refiner'):
            refined_mask = self.mask_refiner(soft_mask)
            # Ensure minimum activation (never completely zero)
            soft_mask = 0.3 + 0.7 * refined_mask
        
        return soft_mask
    
    def forward(self, features, blade_mask=None):
        # Align all features to same size
        target_size = features[0].shape[-2:]
        aligned = []
        
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(feat)
        
        # Concatenate multi-scale features
        concat = torch.cat(aligned, dim=1)
        
        # Apply blade mask with soft gating or attention
        if blade_mask is not None and self.use_soft_gating:
            # Resize blade mask to feature size
            blade_mask_resized = F.interpolate(
                blade_mask.float(),
                size=concat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # Create soft mask with boundary preservation
            soft_mask = self.create_soft_mask(blade_mask_resized)
            
            # Apply blade-guided attention
            attended_features, spatial_att = self.blade_attention(concat, soft_mask)
            
            # Weighted combination of gated and ungated features
            gated_features = attended_features * self.gate_weight
            ungated_features = concat * (1 - self.gate_weight) * 0.3  # Reduced weight for ungated
            concat = gated_features + ungated_features
            
        elif blade_mask is not None:
            # Simple hard masking (fallback)
            blade_mask_resized = F.interpolate(
                blade_mask.float(),
                size=concat.shape[-2:],
                mode='nearest'
            )
            if blade_mask_resized.dim() == 3:
                blade_mask_resized = blade_mask_resized.unsqueeze(1)
            concat = concat * blade_mask_resized
        
        # Feature fusion
        fused = self.seg_fusion(concat)
        
        # Boundary enhancement
        boundary_weight = self.boundary_enhancer(fused)
        fused = fused * (1 + boundary_weight * 0.5)  # Enhance boundary regions
        
        # Segmentation output
        seg_output = self.seg_head(fused)
        seg_output = F.interpolate(seg_output, size=(640, 640), mode='bilinear', align_corners=False)
        
        # Multi-label classification
        multilabel = self.multilabel_head(fused)
        
        return {
            'segmentation': seg_output,
            'multilabel': torch.sigmoid(multilabel)
        }
    
    def get_attention_maps(self, features, blade_mask):
        """어텐션 맵 시각화용"""
        if not self.use_soft_gating:
            return None
            
        with torch.no_grad():
            blade_mask_resized = F.interpolate(
                blade_mask.float(),
                size=features[0].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            soft_mask = self.create_soft_mask(blade_mask_resized)
            concat = torch.cat([F.interpolate(f, size=features[0].shape[-2:], mode='bilinear') 
                               for f in features], dim=1)
            _, spatial_att = self.blade_attention(concat, soft_mask)
            
            return {
                'soft_mask': soft_mask,
                'spatial_attention': spatial_att,
                'boundary_weight': self.boundary_enhancer(self.seg_fusion(concat))
            }