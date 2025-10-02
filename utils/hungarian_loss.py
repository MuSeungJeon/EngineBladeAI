# utils/hungarian_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple


class HungarianLoss(nn.Module):
    """
    Hungarian Loss for Mask2Former with Gaussian constraint support
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        matcher_cost_class: float = 2.0,
        matcher_cost_mask: float = 5.0,
        matcher_cost_dice: float = 5.0,
        no_object_weight: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.matcher_cost_class = matcher_cost_class
        self.matcher_cost_mask = matcher_cost_mask
        self.matcher_cost_dice = matcher_cost_dice
        self.no_object_weight = no_object_weight
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Hungarian loss
        
        Args:
            outputs: Model outputs with keys:
                - pred_logits: [B, Q, C] class predictions
                - pred_masks: [B, Q, H, W] mask predictions
                - aux_outputs: auxiliary outputs from decoder layers
            targets: Ground truth with keys:
                - labels: class labels
                - masks: ground truth masks
                - multilabel: [B, C] multi-label annotations
        """
        
        # Get predictions
        pred_logits = outputs['pred_logits']  # [B, Q, C]
        pred_masks = outputs['pred_masks']    # [B, Q, H, W]
        
        batch_size = pred_logits.shape[0]
        device = pred_logits.device
        
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # Process each batch item
        for b in range(batch_size):
            # Get targets for this batch
            if 'instance_masks' in targets and targets['instance_masks'] is not None:
                # Instance-level supervision
                loss = self._compute_instance_loss(
                    pred_logits[b],
                    pred_masks[b],
                    targets['instance_masks'][b],
                    targets.get('instance_labels', targets.get('labels'))[b]
                )
            elif 'multilabel' in targets:
                # Multi-label supervision (pseudo instances)
                loss = self._compute_multilabel_loss(
                    pred_logits[b],
                    pred_masks[b],
                    targets['multilabel'][b],
                    targets.get('damage_mask', targets.get('masks'))[b]
                )
            else:
                # Fallback: simple loss
                loss = self._compute_simple_loss(
                    pred_logits[b],
                    pred_masks[b],
                    targets
                )
            
            total_loss = total_loss + loss
        
        # Average over batch
        total_loss = total_loss / batch_size
        
        # Add auxiliary losses if present
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_loss = self.forward(
                    {'pred_logits': aux_outputs['pred_logits'],
                     'pred_masks': aux_outputs['pred_masks']},
                    targets
                )
                total_loss = total_loss + aux_loss * 0.1  # Auxiliary weight
        
        return total_loss
    
    def _compute_instance_loss(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        gt_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with instance-level annotations"""
        
        num_queries = pred_logits.shape[0]
        
        # Handle case where gt_masks is a list
        if isinstance(gt_masks, list):
            if len(gt_masks) == 0:
                # No instances - all queries should predict background
                return F.cross_entropy(
                    pred_logits,
                    torch.full((num_queries,), self.num_classes, device=pred_logits.device)
                ) * self.no_object_weight
            gt_masks = torch.stack(gt_masks)
            
        num_instances = gt_masks.shape[0]
        
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(
            pred_logits, pred_masks, gt_masks, gt_labels
        )
        
        # Hungarian matching
        indices = linear_sum_assignment(cost_matrix.cpu().numpy())
        matched_queries = indices[0]
        matched_instances = indices[1]
        
        # Compute losses for matched pairs
        total_loss = 0
        
        # Classification loss
        class_targets = torch.full(
            (num_queries,), self.num_classes, device=pred_logits.device
        )
        class_targets[matched_queries] = gt_labels[matched_instances]
        class_loss = F.cross_entropy(pred_logits, class_targets, reduction='mean')
        
        # Mask loss (only for matched queries)
        mask_loss = 0
        dice_loss = 0
        for q, inst in zip(matched_queries, matched_instances):
            # Resize gt_mask if needed
            if pred_masks[q].shape != gt_masks[inst].shape:
                gt_mask_resized = F.interpolate(
                    gt_masks[inst].unsqueeze(0).unsqueeze(0).float(),
                    size=pred_masks[q].shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            else:
                gt_mask_resized = gt_masks[inst].float()
            
            # BCE loss
            mask_loss += F.binary_cross_entropy_with_logits(
                pred_masks[q].flatten(),
                gt_mask_resized.flatten()
            )
            
            # Dice loss
            pred_sigmoid = pred_masks[q].sigmoid().flatten()
            gt_flat = gt_mask_resized.flatten()
            dice_loss += 1 - (2 * (pred_sigmoid * gt_flat).sum() + 1) / \
                            (pred_sigmoid.sum() + gt_flat.sum() + 1)
        
        if len(matched_queries) > 0:
            mask_loss = mask_loss / len(matched_queries)
            dice_loss = dice_loss / len(matched_queries)
        
        total_loss = class_loss + mask_loss * 5.0 + dice_loss * 2.0
        
        return total_loss
    
    def _compute_multilabel_loss(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        multilabel: torch.Tensor,
        damage_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss with multi-label annotations"""
        
        num_queries = pred_logits.shape[0]
        device = pred_logits.device
        
        # Find active classes
        active_classes = torch.where(multilabel > 0)[0]
        
        if len(active_classes) == 0:
            # No damage - all queries predict no-object
            return F.cross_entropy(
                pred_logits,
                torch.full((num_queries,), self.num_classes, device=device)
            ) * self.no_object_weight
        
        # Assign queries to classes (simple division)
        queries_per_class = num_queries // len(active_classes)
        
        total_loss = 0
        
        for i, cls_idx in enumerate(active_classes):
            start_q = i * queries_per_class
            end_q = (i + 1) * queries_per_class if i < len(active_classes) - 1 else num_queries
            
            # Classification loss for assigned queries
            class_targets = torch.full(
                (end_q - start_q,), cls_idx.item(), device=device
            )
            class_loss = F.cross_entropy(
                pred_logits[start_q:end_q],
                class_targets,
                reduction='mean'
            )
            
            # Mask loss (if damage mask available)
            if damage_mask is not None:
                mask_loss = F.binary_cross_entropy_with_logits(
                    pred_masks[start_q:end_q].mean(0).flatten(),
                    damage_mask.float().flatten()
                )
                total_loss += class_loss + mask_loss
            else:
                total_loss += class_loss
        
        return total_loss / len(active_classes)
    
    def _compute_simple_loss(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        targets: Dict
    ) -> torch.Tensor:
        """Fallback simple loss computation"""
        
        # Simple classification loss
        if 'labels' in targets:
            class_loss = F.cross_entropy(
                pred_logits.mean(0, keepdim=True),
                targets['labels']
            )
        else:
            class_loss = torch.tensor(0.0, device=pred_logits.device)
        
        # Simple mask loss
        if 'masks' in targets:
            mask_loss = F.binary_cross_entropy_with_logits(
                pred_masks.mean(0).flatten(),
                targets['masks'].float().flatten()
            )
        else:
            mask_loss = torch.tensor(0.0, device=pred_logits.device)
        
        return class_loss + mask_loss
    
    def _compute_cost_matrix(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        gt_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute cost matrix for Hungarian matching"""
        
        num_queries = pred_logits.shape[0]
        num_instances = gt_masks.shape[0]
        
        # Classification cost
        pred_probs = F.softmax(pred_logits, dim=-1)
        class_cost = -pred_probs[:, gt_labels]  # [Q, N]
        
        # Mask costs
        pred_masks_sigmoid = pred_masks.sigmoid()
        
        # Flatten for computation
        pred_flat = pred_masks_sigmoid.flatten(1)  # [Q, H*W]
        gt_flat = gt_masks.flatten(1).float()      # [N, H*W]
        
        # Focal loss style mask cost
        mask_cost = -(pred_flat[:, None] * gt_flat[None]).sum(-1) / gt_flat.sum(-1)
        
        # Dice cost
        numerator = 2 * (pred_flat[:, None] * gt_flat[None]).sum(-1)
        denominator = pred_flat.sum(-1)[:, None] + gt_flat.sum(-1)[None]
        dice_cost = 1 - numerator / (denominator + 1e-8)
        
        # Total cost matrix
        cost_matrix = (
            self.matcher_cost_class * class_cost +
            self.matcher_cost_mask * mask_cost +
            self.matcher_cost_dice * dice_cost
        )
        
        return cost_matrix