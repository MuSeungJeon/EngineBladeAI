# models/unified/unified_model_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class UnifiedModelV2(nn.Module):
    """
    통합 엔진 블레이드 손상 검출 모델
    - Backbone: ConvNeXt-FPN
    - Head-A: SegFormer (blade segmentation)
    - Head-B: Mask2Former (damage detection)
    - Learnable soft constraint
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 설정
        self.num_classes = config.get('num_classes', 3)  # crack, nick, tear
        self.use_hungarian = config.get('use_hungarian', True)
        
        # Backbone
        from models.backbones.convnext_fpn import ConvNeXtFPN
        self.backbone = ConvNeXtFPN(
            model_name=config.get('backbone_model', 'convnext_tiny'),
            pretrained=config.get('pretrained', True),
            fpn_channels=config.get('fpn_channels', 256)
        )
        
        # Head-A: Blade Segmentation
        from models.heads.segformer_blade_head import SegFormerBladeHead
        self.blade_head = SegFormerBladeHead(
            in_channels=config.get('fpn_channels', 256),
            num_classes=1,  # binary segmentation
            embed_dims=config.get('blade_embed_dims', [64, 128, 256, 512]),
            num_heads=config.get('blade_num_heads', [1, 2, 4, 8]),
            mlp_ratios=config.get('blade_mlp_ratios', [4, 4, 4, 4]),
            dropout_rate=config.get('blade_dropout', 0.1)
        )
        
        # Head-B: Damage Detection
        from models.heads.mask2former_damage_head import Mask2FormerDamageHead
        self.damage_head = Mask2FormerDamageHead(
            in_channels=config.get('fpn_channels', 256),
            num_classes=self.num_classes,
            num_queries=config.get('num_queries', 100),
            dec_layers=config.get('dec_layers', 3),
            hidden_dim=config.get('hidden_dim', 256),
            dim_feedforward=config.get('dim_feedforward', 1024),
            dropout=config.get('damage_dropout', 0.1)
        )
        
        # Learnable soft constraint parameters
        self.base_weight = nn.Parameter(torch.tensor(0.3))
        self.blade_weight = nn.Parameter(torch.tensor(0.7))
        
        # Loss functions
        self._setup_losses(config)
        
    def _setup_losses(self, config):
        """Loss 함수 설정"""
        # Blade loss
        self.blade_loss_fn = nn.BCEWithLogitsLoss()
        
        # Damage loss
        if self.use_hungarian:
            from utils.hungarian_loss import HungarianLoss
            self.damage_loss_fn = HungarianLoss(
                num_classes=self.num_classes,
                matcher_cost_class=config.get('cost_class', 2.0),
                matcher_cost_mask=config.get('cost_mask', 5.0),
                matcher_cost_dice=config.get('cost_dice', 5.0)
            )
        else:
            from utils.simple_loss import SimpleLoss
            self.damage_loss_fn = SimpleLoss(
                num_classes=self.num_classes
            )
        
        # Loss weights
        self.blade_loss_weight = config.get('blade_loss_weight', 1.0)
        self.damage_loss_weight = config.get('damage_loss_weight', 2.0)
    
    def apply_soft_constraint(
        self, 
        features: Dict[str, torch.Tensor], 
        blade_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Learnable soft constraint 적용
        
        Args:
            features: FPN features dict {'P2': ..., 'P3': ..., ...}
            blade_mask: [B, 1, H, W] blade segmentation mask
        
        Returns:
            constrained_features: soft constraint가 적용된 features
        """
        # Sigmoid로 weight를 0~1 범위로 제한
        base_w = torch.sigmoid(self.base_weight)
        blade_w = torch.sigmoid(self.blade_weight)
        
        # Soft constraint mask 생성
        constraint_mask = base_w + blade_w * blade_mask
        
        # 모든 FPN 레벨에 동일하게 적용
        constrained_features = {}
        for level, feat in features.items():
            # Feature 크기에 맞게 resize
            h, w = feat.shape[-2:]
            mask_resized = F.interpolate(
                constraint_mask,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
            # Constraint 적용
            constrained_features[level] = feat * mask_resized
            
        return constrained_features
    
    def forward(
        self, 
        x: torch.Tensor,
        targets: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
            targets: Ground truth (training only)
        
        Returns:
            outputs: predictions and losses (if training)
        """
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Head-A: Blade segmentation
        blade_logits = self.blade_head(features)  # [B, 1, H, W]
        blade_mask = torch.sigmoid(blade_logits)
        
        # Soft constraint 적용 (학습/추론 모두)
        constrained_features = self.apply_soft_constraint(features, blade_mask)
        
        # Head-B: Damage detection with constrained features
        damage_outputs = self.damage_head(constrained_features)
        
        # 결과 정리
        outputs = {
            'blade_logits': blade_logits,
            'blade_mask': blade_mask,
            'pred_logits': damage_outputs['pred_logits'],  # [B, Q, C]
            'pred_masks': damage_outputs['pred_masks'],     # [B, Q, H, W]
        }
        
        # multilabel 예측 (query aggregation)
        if 'multilabel' in damage_outputs:
            outputs['multilabel'] = damage_outputs['multilabel']
        
        # Loss 계산 (training only)
        if self.training and targets is not None:
            losses = self.compute_losses(outputs, targets)
            outputs['losses'] = losses
            
        return outputs
    
    def compute_losses(
        self, 
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Loss 계산"""
        losses = {}
        
        # Blade loss
        if 'blade_mask' in targets:
            blade_loss = self.blade_loss_fn(
                outputs['blade_logits'].squeeze(1),
                targets['blade_mask'].float()
            )
            losses['blade_loss'] = blade_loss * self.blade_loss_weight
        
        # Damage loss
        damage_targets = {
            'labels': targets.get('labels'),
            'masks': targets.get('masks'),
            'instance_masks': targets.get('instance_masks'),
            'multilabel': targets.get('multilabel')
        }
        
        if self.use_hungarian:
            damage_loss = self.damage_loss_fn(
                outputs, 
                damage_targets
            )
        else:
            damage_loss = self.damage_loss_fn(
                outputs,
                damage_targets
            )
        
        losses['damage_loss'] = damage_loss * self.damage_loss_weight
        
        # Total loss
        losses['total_loss'] = losses.get('blade_loss', 0) + losses.get('damage_loss', 0)
        
        # Constraint weights 모니터링용
        losses['base_weight'] = torch.sigmoid(self.base_weight).detach()
        losses['blade_weight'] = torch.sigmoid(self.blade_weight).detach()
        
        return losses
    
    def get_param_groups(self, base_lr: float = 1e-4) -> List[Dict]:
        """학습률 그룹 설정"""
        return [
            {'params': self.backbone.parameters(), 'lr': base_lr * 0.1},
            {'params': self.blade_head.parameters(), 'lr': base_lr},
            {'params': self.damage_head.parameters(), 'lr': base_lr},
            {'params': [self.base_weight, self.blade_weight], 'lr': base_lr * 0.01}
        ]