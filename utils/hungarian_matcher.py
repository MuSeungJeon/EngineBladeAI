# utils/hungarian_matcher.py (또는 loss 계산하는 곳)
import torch
import torch.nn as nn
import torch.nn.functional as F

class HungarianMatcherFixed(nn.Module):
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        indices = []
        for b in range(bs):
            # instance_masks는 리스트로 오는 경우 처리
            if isinstance(targets, dict):
                tgt_mask = targets['instance_masks'][b]  # [num_inst, 640, 640]
            else:
                tgt_mask = targets[b]['masks']
                
            if len(tgt_mask) == 0:
                indices.append(([], []))
                continue
                
            out_mask = outputs['pred_masks'][b]  # [queries, H, W]
            
            # 크기 확인 및 조정
            out_h, out_w = out_mask.shape[-2:]
            tgt_h, tgt_w = tgt_mask.shape[-2:]
            
            if (out_h, out_w) != (tgt_h, tgt_w):
                # pred_masks를 target 크기로 업샘플링
                out_mask = F.interpolate(
                    out_mask.unsqueeze(0),  # [1, queries, h, w]
                    size=(tgt_h, tgt_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Cost 계산
            out_mask_flat = out_mask.flatten(1).sigmoid()
            tgt_mask_flat = tgt_mask.float().flatten(1)
            
            cost_mask = torch.cdist(out_mask_flat, tgt_mask_flat, p=1)
            
            # Hungarian algorithm
            from scipy.optimize import linear_sum_assignment
            indices_i = linear_sum_assignment(cost_mask.cpu().numpy())
            indices.append(indices_i)
            
        return indices