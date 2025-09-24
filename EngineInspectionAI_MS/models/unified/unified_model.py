# models/unified/unified_model.py
import torch
import torch.nn as nn
from pathlib import Path

from models.backbones.convnext_fpn import ConvNeXtFPN
from models.heads.blade_head import SegFormerBladeHead
from models.heads.mask2former_damage_head import Mask2FormerDamageHead
from models.heads.damage_head import DamageDetectionHead  # CNN 헤드도 import


class UnifiedModel(nn.Module):
    """통합 모델: 백본 + Head-A + Head-B"""
    
    def __init__(
        self,
        backbone_type='tiny',
        num_blade_classes=2,
        num_damage_classes=3,
        pretrained_backbone=True,
        blade_checkpoint=None,
        freeze_blade=True,
        use_fpn=True,
        damage_head_type='mask2former',  # 추가: 'cnn' or 'mask2former'
        damage_head_config=None  # 추가: 헤드별 설정
    ):
        super().__init__()
        
        # Shared backbone
        self.backbone = ConvNeXtFPN(
            model_name=backbone_type,
            pretrained=pretrained_backbone,
            fpn_channels=256,
            use_fpn=use_fpn
        )
        
        # Head-A: Blade detection (변경 없음)
        self.blade_head = SegFormerBladeHead(
            in_channels=256,
            num_classes=num_blade_classes
        )
        
        # Head-B: Damage detection (타입 선택 가능)
        self.damage_head_type = damage_head_type
        
        if damage_head_type == 'mask2former':
            # Mask2Former 설정
            config = damage_head_config or {}
            self.damage_head = Mask2FormerDamageHead(
                in_channels=256,
                num_classes=num_damage_classes,
                num_queries=config.get('num_queries', 200),
                hidden_dim=config.get('hidden_dim', 256),
                num_heads=config.get('num_heads', 8),
                dec_layers=config.get('dec_layers', 6),
                dropout=config.get('dropout', 0.1),
                use_blade_mask=True
            )
        else:
            # CNN 헤드
            config = damage_head_config or {}
            self.damage_head = DamageDetectionHead(
                in_channels=256,
                num_classes=num_damage_classes,
                use_soft_gating=config.get('use_soft_gating', True),
                boundary_margin=config.get('boundary_margin', 10)
            )
        
        # Load pretrained blade head if provided
        if blade_checkpoint and Path(blade_checkpoint).exists():
            self._load_blade_checkpoint(blade_checkpoint)
        
        # Freeze blade head if specified
        if freeze_blade:
            for param in self.blade_head.parameters():
                param.requires_grad = False
    
    def forward(self, x, use_blade_mask=True):
        # Extract features
        features = self.backbone(x)
        
        # Blade detection
        blade_output = self.blade_head(features)
        
        # Generate blade mask
        blade_mask = None
        if use_blade_mask:
            blade_mask = (blade_output.argmax(1, keepdim=True) == 1).float()
        
        # Damage detection with blade mask
        if self.damage_head_type == 'mask2former':
            # Mask2Former는 features 리스트를 받음
            damage_output = self.damage_head(features, blade_mask)
        else:
            # CNN은 단일 feature를 받음
            damage_output = self.damage_head(features, blade_mask)
        
        return {
            'blade': blade_output,
            'blade_mask': blade_mask,
            **damage_output
        }
    
    # 나머지 메서드는 그대로 유지
    def _load_blade_checkpoint(self, checkpoint_path):
        """Load pretrained blade head weights"""
        checkpoint = torch.load(checkpoint_path)
        
        blade_state = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'blade_head' in k:
                new_key = k.replace('blade_head.', '')
                blade_state[new_key] = v
        
        if blade_state:
            self.blade_head.load_state_dict(blade_state, strict=False)
            print(f"✅ Loaded blade head from {checkpoint_path}")
    
    def get_trainable_params(self):
        """Get only trainable parameters"""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True