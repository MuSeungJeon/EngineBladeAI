import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .hungarian_matcher import HungarianMatcher

# ==============================================================================
# 헬퍼 함수들 (클래스 바깥에 위치)
# ==============================================================================

class NestedTensor(object):
    def __init__(self, tensors, mask: torch.Tensor):
        self.tensors = tensors
        self.mask = mask
    def to(self, device):
        self.tensors = self.tensors.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        return self
    def decompose(self):
        return self.tensors, self.mask

def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    if not tensor_list:
        raise ValueError("tensor_list should not be empty")
    
    if tensor_list[0].ndim == 3: # For masks: [C, H, W]
        shapes = [list(t.shape) for t in tensor_list]
        max_c = max(s[0] for s in shapes)
        max_h = max(s[1] for s in shapes)
        max_w = max(s[2] for s in shapes)
        
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros((len(tensor_list), max_c, max_h, max_w), dtype=dtype, device=device)
        mask = torch.ones((len(tensor_list), max_h, max_w), dtype=torch.bool, device=device)
        
        for i, t in enumerate(tensor_list):
            tensor[i, :t.shape[0], :t.shape[1], :t.shape[2]].copy_(t)
            mask[i, :t.shape[1], :t.shape[2]] = False
            
    elif tensor_list[0].ndim == 4: # For batched masks: [N, C, H, W]
        # This part might need adjustment based on exact use case
        return tensor_list # Or handle padding for 4D tensors
    else:
        raise ValueError('Unsupported tensor dimension')
    return NestedTensor(tensor, mask)

def dice_loss(inputs, targets, num_boxes):
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum() / num_boxes

# ==============================================================================
# SetCriterion 클래스
# ==============================================================================

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, class_weights):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Focal Loss)."""
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = F.one_hot(target_classes, num_classes=self.num_classes + 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1].float()

        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes=num_boxes,
            alpha=0.25,
            gamma=2.0
        )
        return {'loss_ce': loss_ce}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss."""
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        src_masks = F.interpolate(
            src_masks.unsqueeze(1),
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        losses = {
            "loss_mask": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {'labels': self.loss_labels, 'masks': self.loss_masks}
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses