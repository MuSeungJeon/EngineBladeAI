# utils/hungarian_matcher.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict

class HungarianMatcher(nn.Module):
    """
    Class-wise Hungarian Matcher for Mask2Former.

    This matcher implements a 'class-wise matching' strategy where each class-specific
    group of queries is independently matched against ground truth masks of that class.
    This approach is designed for multi-label datasets where instance-level annotations
    are not available.

    Core Logic:
    1. For each class (crack, nick, tear), select the dedicated 100 queries.
    2. For each image in the batch, check if it contains a ground truth for that class.
    3. If it does, compute the cost between the 100 queries and the single ground truth mask.
    4. Find the single best-matching query (the one with the minimum cost).
    5. Collect all matched pairs (query_idx, target_idx) for each image.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_classes: int = 3  # crack, nick, tear
    ):
        """
        Args:
            cost_class (float): Relative weight of the classification cost.
            cost_mask (float): Relative weight of the L1 mask cost.
            cost_dice (float): Relative weight of the Dice mask cost.
            num_classes (int): Number of object classes.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_classes = num_classes
        assert self.cost_class != 0 or self.cost_mask != 0 or self.cost_dice != 0, "All costs cannot be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bs, num_queries = outputs["pred_logits"].shape[:2]
        queries_per_class = num_queries // self.num_classes

        indices = [([], []) for _ in range(bs)]

        for class_idx in range(self.num_classes):
            q_start = class_idx * queries_per_class
            out_logits_c = outputs['pred_logits'][:, q_start:q_start + queries_per_class]
            out_masks_c = outputs['pred_masks'][:, q_start:q_start + queries_per_class]

            for b in range(bs):
                currentTarget = targets[b]
                
                if currentTarget['multilabel'][class_idx] == 1:
                    # 해당 클래스에 대한 정답 마스크와 레이블 준비
                    try:
                        target_instance_idx = currentTarget['labels'].tolist().index(class_idx)
                        tgt_mask = currentTarget['masks'][target_instance_idx].unsqueeze(0)
                    except (ValueError, IndexError):
                        continue # 해당 클래스 레이블이 없는 경우 건너뛰기

                    tgt_label = torch.tensor([class_idx], device=out_logits_c.device)
                    
                    aligned_out_masks = self._align_shapes(out_masks_c[b], tgt_mask)
                    cost_class = self._compute_class_cost(out_logits_c[b], tgt_label)
                    cost_mask = self._compute_mask_cost(aligned_out_masks.float(), tgt_mask.float())
                    cost_dice = self._compute_dice_cost(aligned_out_masks.float(), tgt_mask.float())
                    
                    C = (self.cost_class * cost_class +
                         self.cost_mask * cost_mask +
                         self.cost_dice * cost_dice)

                    # --- [핵심 수정] ---
                    # 쿼리 차원(dim=0)을 기준으로 최소값 인덱스를 찾음
                    query_idx = C.argmin(dim=0) # shape: [1]
                    original_query_idx = query_idx.item() + q_start
                    
                    indices[b][0].append(original_query_idx)
                    indices[b][1].append(target_instance_idx)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def _align_shapes(self, out_masks: torch.Tensor, tgt_masks: torch.Tensor) -> torch.Tensor:
        """Align prediction masks to target mask shape via interpolation."""
        if out_masks.shape[-2:] != tgt_masks.shape[-2:]:
            out_masks = F.interpolate(
                out_masks.unsqueeze(0),
                size=tgt_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        return out_masks

    def _compute_class_cost(self, out_logits: torch.Tensor, tgt_labels: torch.Tensor) -> torch.Tensor:
        """Compute classification cost using a focal loss-like approach."""
        out_prob = out_logits.sigmoid()  # [num_queries, num_classes]
        tgt_onehot = F.one_hot(tgt_labels, self.num_classes).float()  # [num_targets, num_classes]

        alpha = 0.25
        gamma = 2.0
        neg_cost = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = torch.einsum('qc,tc->qt', pos_cost, tgt_onehot) + torch.einsum('qc,tc->qt', neg_cost, 1-tgt_onehot)
        return cost_class

    def _compute_mask_cost(self, out_masks: torch.Tensor, tgt_masks: torch.Tensor) -> torch.Tensor:
        """Compute mask cost as L1 distance."""
        out_masks_flat = out_masks.flatten(1).sigmoid()
        tgt_masks_flat = tgt_masks.flatten(1).float()
        
        # --- [핵심 수정] ---
        # cdist가 'Half' (float16) 타입을 지원하지 않으므로, float32로 명시적 변환
        cost_mask = torch.cdist(out_masks_flat.float(), tgt_masks_flat.float(), p=1)
        
        return cost_mask

    def _compute_dice_cost(self, out_masks: torch.Tensor, tgt_masks: torch.Tensor) -> torch.Tensor:
        """Compute mask cost using Dice coefficient."""
        out_masks_flat = out_masks.flatten(1).sigmoid()
        tgt_masks_flat = tgt_masks.flatten(1).float()
        
        # --- [핵심 수정] ---
        # einsum이 'Half' (float16) 타입을 지원하지 않으므로, float32로 명시적 변환
        numerator = 2 * torch.einsum('qh,th->qt', out_masks_flat.float(), tgt_masks_flat.float())
        denominator = out_masks_flat.sum(1)[:, None] + tgt_masks_flat.sum(1)[None, :]
        
        cost_dice = 1 - (numerator + 1e-8) / (denominator + 1e-8)
        return cost_dice