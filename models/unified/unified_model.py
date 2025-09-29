# models/unified/unified_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.backbones.convnext_fpn import ConvNeXtFPN
from models.heads.segformer_blade_head import SegFormerBladeHead
from models.heads.mask2former_damage_head import Mask2FormerDamageHead
from utils.hungarian_loss import HungarianLoss


class UnifiedModel(nn.Module):
    """
    통합 엔진 블레이드 손상 검출 모델
    - Backbone: ConvNeXt-FPN
    - Head-A: SegFormer (blade segmentation)
    - Head-B: Mask2Former with Gaussian constraint (damage detection)
    - 300 queries (100 per class)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.config = config
        self.num_classes = config.get('num_damage_classes', 3)  # crack, nick, tear
        self.use_hungarian = config.get('use_hungarian', False)  # Start with SimpleLoss
        
        # Backbone
        self.backbone = ConvNeXtFPN(
            model_name=config.get('backbone_type', 'tiny'),  # 'tiny', 'small', 'base'
            pretrained=config.get('pretrained', True),
            fpn_channels=config.get('fpn_channels', 256)
        )
        
        # Head-A: Blade Segmentation (SegFormer)
        blade_config = config.get('blade_head', {})
        self.blade_head = SegFormerBladeHead(
            in_channels=config.get('fpn_channels', 256),
            num_classes=1,  # binary segmentation
            embed_dims=blade_config.get('embed_dims', [64, 128, 256, 512]),
            num_heads=blade_config.get('num_heads', [1, 2, 4, 8]),
            mlp_ratios=blade_config.get('mlp_ratios', [4, 4, 4, 4]),
            dropout_rate=blade_config.get('dropout_rate', 0.1)
        )
        
        # Head-B: Damage Detection (Mask2Former with 300 queries)
        damage_config = config.get('mask2former_config', {})
        self.damage_head = Mask2FormerDamageHead(
            in_channels=config.get('fpn_channels', 256),
            num_classes=self.num_classes,
            queries_per_class=damage_config.get('queries_per_class', 100),  # 클래스당 100개
            hidden_dim=damage_config.get('hidden_dim', 256),
            num_heads=damage_config.get('num_heads', 8),
            dim_feedforward=damage_config.get('dim_feedforward', 1024),
            dec_layers=damage_config.get('dec_layers', 3),
            dropout=damage_config.get('dropout', 0.1),
            use_blade_mask=damage_config.get('use_blade_mask', True)
        )
        
        # Loss functions
        self._setup_losses()
        
        # Loss weights
        self.blade_loss_weight = config.get('blade_loss_weight', 1.0)
        self.damage_loss_weight = config.get('damage_loss_weight', 2.0)
        self.aux_loss_weight = config.get('aux_loss_weight', 0.4)
        
    def _setup_losses(self):
        """Setup loss functions"""
        
        # Blade segmentation loss
        self.blade_criterion = nn.BCEWithLogitsLoss()
        
        # Damage detection loss
        if self.use_hungarian:
            # Hungarian matching for instance-level supervision
            from utils.hungarian_loss import HungarianLoss
            self.damage_criterion = HungarianLoss(
                num_classes=self.num_classes,
                cost_class=2.0,
                cost_mask=5.0,
                cost_dice=5.0
            )
        else:
            # Simple loss for multi-label classification
            self.damage_criterion = SimpleDamageLoss(
                num_classes=self.num_classes
            )
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W]
            targets: Optional dict with:
                - blade_mask: [B, H, W]
                - damage_mask: [B, H, W]
                - multilabel: [B, num_classes]
                - instance_masks: List of instance masks (if using Hungarian)
                - instance_labels: List of instance labels (if using Hungarian)
        
        Returns:
            outputs: Dict with predictions
            losses: Dict with loss values (if targets provided)
        """
        
        batch_size = images.size(0)
        
        # 1. Feature extraction
        features = self.backbone(images)
        
        # 2. Blade segmentation (Head-A)
        blade_logits = self.blade_head(features)  # [B, 1, H, W]
        blade_mask = torch.sigmoid(blade_logits.squeeze(1))  # [B, H, W]
        
        # 3. Damage detection with blade constraint (Head-B)
        damage_outputs = self.damage_head(
            features,
            blade_mask=blade_mask
        )
        
        # 4. Prepare outputs
        outputs = {
            'blade_logits': blade_logits,
            'blade_mask': blade_mask,
            'pred_logits': damage_outputs['pred_logits'],  # [B, 300, 3]
            'pred_masks': damage_outputs['pred_masks'],     # [B, 300, H, W]
            'multilabel': damage_outputs['multilabel'],     # [B, 3]
            'multilabel_alt': damage_outputs.get('multilabel_alt'),  # [B, 3]
            'aux_outputs': damage_outputs.get('aux_outputs', [])
        }
        
        # 5. Calculate losses if targets provided
        losses = {}
        if targets is not None:
            losses = self.compute_losses(outputs, targets)
        
        return outputs, losses
    
    def compute_losses(
        self,
        outputs: Dict,
        targets: Dict
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        
        losses = {}
        
        # 1. Blade segmentation loss
        if 'blade_mask' in targets:
            blade_loss = self.blade_criterion(
                outputs['blade_logits'].squeeze(1),
                targets['blade_mask'].float()
            )
            losses['blade_loss'] = blade_loss * self.blade_loss_weight
        
        # 2. Damage detection loss
        if self.use_hungarian and 'instance_masks' in targets:
            # Hungarian matching loss
            damage_targets = [{
                'labels': targets['instance_labels'][i],
                'masks': targets['instance_masks'][i]
            } for i in range(len(targets['instance_labels']))]
            
            damage_loss = self.damage_criterion(
                outputs, damage_targets
            )
            losses['damage_loss'] = damage_loss * self.damage_loss_weight
            
            # Auxiliary losses for intermediate layers
            if 'aux_outputs' in outputs:
                for i, aux_output in enumerate(outputs['aux_outputs']):
                    aux_loss = self.damage_criterion(aux_output, damage_targets)
                    losses[f'aux_loss_{i}'] = aux_loss * self.aux_loss_weight
                    
        else:
            # Simple multi-label loss
            damage_loss = self.damage_criterion(
                outputs, targets
            )
            losses.update({
                k: v * self.damage_loss_weight 
                for k, v in damage_loss.items()
            })
        
        # 3. Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def get_param_groups(self, base_lr: float = 1e-4):
        """
        Get parameter groups with different learning rates
        
        Returns parameter groups for:
        - Backbone: base_lr * 0.1 (fine-tuning)
        - Blade head: base_lr
        - Damage head main: base_lr
        - Gaussian constraint params: base_lr * 0.01 (slow learning)
        """
        
        param_groups = [
            # Backbone - slower learning rate
            {
                'params': self.backbone.parameters(),
                'lr': base_lr * 0.1,
                'name': 'backbone'
            },
            # Head-A - normal learning rate
            {
                'params': self.blade_head.parameters(),
                'lr': base_lr,
                'name': 'blade_head'
            },
            # Head-B main parameters - normal learning rate
            {
                'params': [p for n, p in self.damage_head.named_parameters() 
                          if not n.startswith(('sigma', 'min_weight', 'blade_weight'))],
                'lr': base_lr,
                'name': 'damage_head'
            },
            # Gaussian constraint parameters - slower learning
            {
                'params': [self.damage_head.sigma, 
                          self.damage_head.min_weight,
                          self.damage_head.blade_weight],
                'lr': base_lr * 0.01,
                'name': 'constraint_params'
            }
        ]
        
        return param_groups
    
    def freeze_backbone(self):
        """Freeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class SimpleDamageLoss(nn.Module):
    """Simple loss for damage detection without Hungarian matching"""
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
    def forward(
        self,
        outputs: Dict,
        targets: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate simple losses
        
        Args:
            outputs: Model outputs with pred_logits, pred_masks, multilabel
            targets: Ground truth with damage_mask, multilabel
        """
        
        losses = {}
        
        # Multi-label classification loss
        if 'multilabel' in outputs and 'multilabel' in targets:
            losses['multilabel_bce'] = self.bce_loss(
                outputs['multilabel'],
                targets['multilabel'].float()
            )
        
        # Mask loss (if damage masks available)
        if 'pred_masks' in outputs and 'damage_mask' in targets:
            # Average over all queries for simple loss
            pred_masks_avg = outputs['pred_masks'].mean(dim=1)  # [B, H, W]
            
            losses['mask_bce'] = self.bce_loss(
                pred_masks_avg,
                targets['damage_mask'].float()
            )
            
            losses['mask_dice'] = self.dice_loss(
                torch.sigmoid(pred_masks_avg),
                targets['damage_mask'].float()
            )
        
        return losses


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        intersection = (pred * target).sum(1)
        dice = (2. * intersection + self.smooth) / (
            pred.sum(1) + target.sum(1) + self.smooth
        )
        
        return 1 - dice.mean()