# utils/hungarian_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hungarian_matcher import HungarianMatcherFixed

class HungarianLoss(nn.Module):
    def __init__(self, num_classes=3, weight_dict=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcherFixed()
        
        # Loss weights
        self.weight_dict = weight_dict or {
            'loss_ce': 2.0,
            'loss_mask': 5.0,
            'loss_dice': 2.0
        }
        
    def forward(self, outputs, batch):
        # Target 형식 맞추기
        targets = {
            'instance_masks': batch['instance_masks'],
            'instance_labels': batch['instance_labels']
        }
        
        # Hungarian matching
        indices = self.matcher(outputs, targets)
        
        # 각 loss 계산
        losses = {}
        losses['ce'] = self.loss_labels(outputs, targets, indices)
        losses['mask'] = self.loss_masks(outputs, targets, indices)
        losses['dice'] = self.loss_dice(outputs, targets, indices)
        
        # Total loss
        total_loss = sum(losses[k] * self.weight_dict[f'loss_{k}'] 
                        for k in losses.keys())
        
        return total_loss, losses
    
    def loss_labels(self, outputs, targets, indices):
        """Classification loss"""
        src_logits = outputs['pred_logits']  # [B, Q, C]
        
        # Matched pairs만 loss 계산
        loss = 0
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
                
            pred = src_logits[i, src_idx]  # [matched_q, C]
            tgt = targets['instance_labels'][i][tgt_idx]  # [matched_q]
            
            loss += F.cross_entropy(pred, tgt)
            
        return loss / len(indices)
    
    def loss_masks(self, outputs, targets, indices):
        """Mask loss (BCE)"""
        src_masks = outputs['pred_masks']
        
        # 크기 맞추기 (이미 matcher에서 했지만 확인)
        if src_masks.shape[-2:] != targets['instance_masks'][0].shape[-2:]:
            src_masks = F.interpolate(
                src_masks,
                size=targets['instance_masks'][0].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        loss = 0
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
                
            pred = src_masks[i, src_idx]
            tgt = targets['instance_masks'][i][tgt_idx].float()
            
            loss += F.binary_cross_entropy_with_logits(
                pred.flatten(1), 
                tgt.flatten(1)
            )
            
        return loss / len(indices)
    
    def loss_dice(self, outputs, targets, indices):
        """Dice loss"""
        src_masks = outputs['pred_masks']
        
        # 크기 맞추기
        if src_masks.shape[-2:] != targets['instance_masks'][0].shape[-2:]:
            src_masks = F.interpolate(
                src_masks,
                size=targets['instance_masks'][0].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        loss = 0
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
                
            pred = src_masks[i, src_idx].sigmoid()
            tgt = targets['instance_masks'][i][tgt_idx].float()
            
            numerator = 2 * (pred * tgt).sum(dim=[1, 2])
            denominator = pred.sum(dim=[1, 2]) + tgt.sum(dim=[1, 2])
            
            loss += (1 - (numerator + 1) / (denominator + 1)).mean()
            
        return loss / len(indices)