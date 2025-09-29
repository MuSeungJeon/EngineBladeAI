# utils/hungarian_matcher.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional

class HungarianMatcherFixed(nn.Module):
    """
    Hungarian Matcher with shape mismatch handling for Mask2Former
    
    Handles:
    1. Shape mismatches between pred_masks and target masks
    2. Instance masks coming as list or dict
    3. Empty target cases
    4. Multi-label to instance conversion
    """
    
    def __init__(
        self, 
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_classes: int = 3  # crack, nick, tear
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_classes = num_classes
        
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> List[Tuple]:
        """
        Args:
            outputs: dict containing:
                - pred_logits: [batch_size, num_queries, num_classes]
                - pred_masks: [batch_size, num_queries, H, W]
            targets: dict containing:
                - multilabel: [batch_size, num_classes] (binary labels)
                - instance_masks: List of [num_instances, H, W] or pseudo instances
                - labels: List of [num_instances] class labels (if available)
        
        Returns:
            List of matched indices for each batch
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        indices = []
        for b in range(bs):
            # Get target masks and labels
            tgt_masks, tgt_labels = self._get_targets(targets, b)
            
            if len(tgt_masks) == 0:
                # No targets in this batch - match nothing
                indices.append((torch.tensor([], dtype=torch.long), 
                              torch.tensor([], dtype=torch.long)))
                continue
            
            # Get predictions for this batch
            out_logits = outputs['pred_logits'][b]  # [num_queries, num_classes]
            out_masks = outputs['pred_masks'][b]    # [num_queries, H, W]
            
            # Handle shape mismatch
            out_masks = self._align_shapes(out_masks, tgt_masks)
            
            # Compute cost matrices
            cost_class = self._compute_class_cost(out_logits, tgt_labels)
            cost_mask = self._compute_mask_cost(out_masks, tgt_masks)
            cost_dice = self._compute_dice_cost(out_masks, tgt_masks)
            
            # Final cost matrix
            C = self.cost_class * cost_class + \
                self.cost_mask * cost_mask + \
                self.cost_dice * cost_dice
            
            # Hungarian assignment
            indices_i = linear_sum_assignment(C.cpu().numpy())
            indices.append((torch.tensor(indices_i[0], dtype=torch.long),
                          torch.tensor(indices_i[1], dtype=torch.long)))
            
        return indices
    
    def _get_targets(self, targets: Dict, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract target masks and labels, handling different formats"""
        
        # Case 1: instance_masks already exist
        if 'instance_masks' in targets:
            if isinstance(targets['instance_masks'], list):
                tgt_masks = targets['instance_masks'][batch_idx]
            else:
                tgt_masks = targets['instance_masks'][batch_idx]
                
            # Get labels if available, otherwise infer from masks
            if 'labels' in targets:
                if isinstance(targets['labels'], list):
                    tgt_labels = targets['labels'][batch_idx]
                else:
                    tgt_labels = targets['labels'][batch_idx]
            else:
                # Infer labels from which masks exist
                tgt_labels = self._infer_labels_from_masks(tgt_masks, targets['multilabel'][batch_idx])
                
        # Case 2: Only multilabel - create pseudo instances
        else:
            tgt_masks, tgt_labels = self._create_pseudo_instances(
                targets['multilabel'][batch_idx],
                targets.get('damage_mask', None)
            )
            
        return tgt_masks, tgt_labels
    
    def _align_shapes(self, out_masks: torch.Tensor, tgt_masks: torch.Tensor) -> torch.Tensor:
        """Align prediction masks to target mask shape"""
        out_h, out_w = out_masks.shape[-2:]
        tgt_h, tgt_w = tgt_masks.shape[-2:]
        
        if (out_h, out_w) != (tgt_h, tgt_w):
            # Interpolate to match target size
            out_masks = F.interpolate(
                out_masks.unsqueeze(0),  # [1, num_queries, H, W]
                size=(tgt_h, tgt_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [num_queries, H, W]
            
        return out_masks
    
    def _compute_class_cost(self, out_logits: torch.Tensor, tgt_labels: torch.Tensor) -> torch.Tensor:
        """Compute classification cost matrix using focal loss style"""
        # Sigmoid for multi-label
        out_prob = out_logits.sigmoid()  # [num_queries, num_classes]
        
        # Create one-hot target
        tgt_onehot = F.one_hot(tgt_labels, self.num_classes).float()  # [num_tgt, num_classes]
        
        # Focal loss parameters
        alpha = 0.25
        gamma = 2.0
        
        # Compute costs
        neg_cost = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        
        # Cost matrix [num_queries, num_targets]
        cost_class = torch.einsum('qc,tc->qt', pos_cost - neg_cost, tgt_onehot)
        
        return cost_class
    
    def _compute_mask_cost(self, out_masks: torch.Tensor, tgt_masks: torch.Tensor) -> torch.Tensor:
        """Compute L1 distance between masks"""
        out_masks_flat = out_masks.flatten(1).sigmoid()  # [num_queries, H*W]
        tgt_masks_flat = tgt_masks.float().flatten(1)    # [num_targets, H*W]
        
        # Ensure float type for cdist
        out_masks_flat = out_masks_flat.float()
        tgt_masks_flat = tgt_masks_flat.float()
        
        # L1 distance
        cost_mask = torch.cdist(out_masks_flat, tgt_masks_flat, p=1) / tgt_masks_flat.shape[1]
        
        return cost_mask
    
    def _compute_dice_cost(self, out_masks: torch.Tensor, tgt_masks: torch.Tensor) -> torch.Tensor:
        """Compute dice coefficient cost"""
        out_masks_flat = out_masks.flatten(1).sigmoid()  # [num_queries, H*W]
        tgt_masks_flat = tgt_masks.float().flatten(1)    # [num_targets, H*W]
        
        # Dice coefficient
        numerator = 2 * torch.einsum('qh,th->qt', out_masks_flat, tgt_masks_flat)
        denominator = out_masks_flat.sum(1)[:, None] + tgt_masks_flat.sum(1)[None, :]
        cost_dice = 1 - (numerator + 1) / (denominator + 1)
        
        return cost_dice
    
    def _create_pseudo_instances(
        self, 
        multilabel: torch.Tensor,
        damage_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create pseudo instance masks from multi-label annotation
        
        This is a workaround when we don't have instance-level annotations
        """
        device = multilabel.device
        active_classes = torch.where(multilabel > 0)[0]
        
        if len(active_classes) == 0:
            # No damage in this image
            return torch.zeros((0, 640, 640), device=device), torch.zeros((0,), dtype=torch.long, device=device)
        
        # Create one pseudo instance per active class
        pseudo_masks = []
        labels = []
        
        for class_idx in active_classes:
            if damage_mask is not None:
                # Use the damage mask for this class
                pseudo_masks.append(damage_mask)
            else:
                # Create a dummy mask (this is not ideal but necessary without instance annotations)
                # In practice, you'd want to use some heuristic or model to generate these
                dummy_mask = torch.ones((640, 640), device=device) * 0.1  # Small activation everywhere
                pseudo_masks.append(dummy_mask)
            
            labels.append(class_idx)
        
        return torch.stack(pseudo_masks), torch.tensor(labels, device=device)
    
    def _infer_labels_from_masks(
        self, 
        instance_masks: torch.Tensor,
        multilabel: torch.Tensor
    ) -> torch.Tensor:
        """Infer instance labels from multi-label annotation"""
        num_instances = len(instance_masks)
        active_classes = torch.where(multilabel > 0)[0]
        
        if num_instances == 0:
            return torch.zeros((0,), dtype=torch.long, device=instance_masks.device)
        
        # Simple heuristic: assign classes cyclically to instances
        # In practice, you might want a smarter assignment based on mask properties
        labels = []
        for i in range(num_instances):
            if i < len(active_classes):
                labels.append(active_classes[i])
            else:
                # More instances than active classes - repeat
                labels.append(active_classes[i % len(active_classes)])
                
        return torch.tensor(labels, device=instance_masks.device)