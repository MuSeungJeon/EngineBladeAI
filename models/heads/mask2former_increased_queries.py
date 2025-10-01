# models/heads/mask2former_increased_queries.py
"""
Query 수를 늘리고 클래스별로 특화시킨 Mask2Former
각 클래스당 충분한 query를 할당하여 복잡한 손상 패턴 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class Mask2FormerIncreasedQueries(nn.Module):
    """
    Query를 늘리고 클래스별로 특화시킨 Mask2Former
    
    예: 300 queries = 각 클래스당 100개
    - Query 0~99: Crack 전문가
    - Query 100~199: Nick 전문가  
    - Query 200~299: Tear 전문가
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        queries_per_class: int = 100,  # 클래스당 query 수
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dec_layers: int = 3,
        dropout: float = 0.1,
        pre_norm: bool = False,
        enforce_class_specialization: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.num_queries = num_classes * queries_per_class  # 총 query 수
        self.hidden_dim = hidden_dim
        self.enforce_class_specialization = enforce_class_specialization
        
        # Query embeddings (늘어난 수만큼)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        
        # Query-to-class 매핑
        self.query_to_class = {}
        for q in range(self.num_queries):
            self.query_to_class[q] = q // queries_per_class
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Transformer decoder (메모리 효율적인 설정)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            normalize_first=pre_norm,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=dec_layers
        )
        
        # Output heads
        self.class_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Class specialization mask (각 query가 자신의 클래스만 예측)
        if enforce_class_specialization:
            self.register_buffer('class_mask', self._create_class_mask())
    
    def _create_class_mask(self):
        """각 query가 전문 클래스만 높게 예측하도록 하는 마스크"""
        mask = torch.zeros(self.num_queries, self.num_classes)
        
        for q in range(self.num_queries):
            assigned_class = self.query_to_class[q]
            mask[q, assigned_class] = 1.0  # 해당 클래스 활성화
            
            # 다른 클래스는 억제 (하지만 완전히 막지는 않음)
            for c in range(self.num_classes):
                if c != assigned_class:
                    mask[q, c] = -5.0  # 음수로 억제
        
        return mask
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with increased queries
        
        Args:
            features: FPN features dict
            
        Returns:
            pred_logits: [B, num_queries, num_classes]
            pred_masks: [B, num_queries, H, W]
        """
        
        # Use P4 feature (1/16 scale) as main feature
        if 'P4' in features:
            src = features['P4']
        else:
            src = list(features.values())[2]  # Fallback
        
        batch_size = src.shape[0]
        
        # Input projection
        src_proj = self.input_proj(src)  # [B, hidden_dim, H, W]
        h, w = src_proj.shape[-2:]
        
        # Flatten for transformer
        src_flat = src_proj.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]
        
        # Query embeddings
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer decoder (메모리 효율을 위해 체크포인팅 사용 가능)
        if self.training and self.num_queries > 200:
            # Gradient checkpointing for memory efficiency
            import torch.utils.checkpoint as checkpoint
            hs = checkpoint.checkpoint(
                self.transformer_decoder,
                query_embeds,
                src_flat
            )
        else:
            hs = self.transformer_decoder(query_embeds, src_flat)
        
        # Output projections
        outputs_class = self.class_embed(hs)  # [B, Q, C]
        
        # Class specialization 적용
        if self.enforce_class_specialization:
            # 각 query의 logits에 class mask 적용
            outputs_class = outputs_class + self.class_mask.unsqueeze(0)
        
        # Mask prediction
        mask_embeds = self.mask_embed(hs)  # [B, Q, hidden_dim]
        
        # Dot product with spatial features to get masks
        # [B, Q, hidden_dim] x [B, hidden_dim, H*W] -> [B, Q, H*W]
        outputs_mask = torch.bmm(mask_embeds, src_flat.permute(0, 2, 1))
        outputs_mask = outputs_mask.view(batch_size, self.num_queries, h, w)
        
        # Upsample masks to original size
        outputs_mask = F.interpolate(
            outputs_mask,
            size=(src.shape[-2] * 4, src.shape[-1] * 4),  # 4x upsampling
            mode='bilinear',
            align_corners=False
        )
        
        return {
            'pred_logits': outputs_class,
            'pred_masks': outputs_mask,
            'query_embeds': hs,  # For analysis
            'query_classes': [self.query_to_class[q] for q in range(self.num_queries)]
        }


class OptimizedMultiLabelHungarianLoss(nn.Module):
    """
    늘어난 Query에 최적화된 Hungarian Loss
    메모리 효율적인 매칭
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        queries_per_class: int = 100,
        samples_per_class: int = 10,  # 각 클래스에서 샘플링할 query 수
        weight_dict: Optional[Dict] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.samples_per_class = samples_per_class
        self.weight_dict = weight_dict or {
            'loss_ce': 2.0,
            'loss_mask': 5.0,
            'loss_dice': 2.0
        }
    
    def forward(self, outputs: Dict, targets: Dict):
        """
        메모리 효율적인 Loss 계산
        모든 query를 사용하지 않고 샘플링
        """
        
        batch_size = outputs['pred_logits'].shape[0]
        device = outputs['pred_logits'].device
        
        total_loss = 0
        losses = {}
        
        # 각 배치 아이템 처리
        for b in range(batch_size):
            multilabel = targets['multilabel'][b]
            active_classes = torch.where(multilabel > 0)[0]
            
            if len(active_classes) == 0:
                continue
            
            # damage mask 가져오기
            if 'damage_mask' in targets:
                damage_mask = targets['damage_mask'][b]
            elif 'blade_mask' in targets:
                damage_mask = targets['blade_mask'][b]
                if len(damage_mask.shape) == 3:
                    damage_mask = damage_mask[0]
            else:
                damage_mask = torch.ones((640, 640), device=device) * 0.1
            
            batch_loss = 0
            
            # 각 활성 클래스에 대해
            for cls_id in active_classes:
                # 해당 클래스의 query 범위
                query_start = cls_id * self.queries_per_class
                query_end = (cls_id + 1) * self.queries_per_class
                
                # 해당 클래스 query들의 예측
                class_queries = range(query_start, min(query_end, outputs['pred_logits'].shape[1]))
                
                # 샘플링 (메모리 절약)
                if len(class_queries) > self.samples_per_class:
                    # IoU 기반으로 top-k 선택
                    with torch.no_grad():
                        query_masks = outputs['pred_masks'][b, class_queries].sigmoid()
                        
                        # 크기 맞추기
                        if query_masks.shape[-2:] != damage_mask.shape:
                            query_masks = F.interpolate(
                                query_masks.unsqueeze(0),
                                size=damage_mask.shape,
                                mode='bilinear'
                            ).squeeze(0)
                        
                        # IoU 계산
                        intersection = (query_masks * damage_mask).sum(dim=(1,2))
                        union = query_masks.sum(dim=(1,2)) + damage_mask.sum() - intersection
                        iou = intersection / (union + 1e-6)
                        
                        # Top-k 선택
                        _, top_indices = torch.topk(iou, min(self.samples_per_class, len(iou)))
                        selected_queries = [class_queries[i] for i in top_indices]
                else:
                    selected_queries = list(class_queries)
                
                # 선택된 query들에 대한 loss
                for q_idx in selected_queries:
                    # Classification loss
                    pred_logit = outputs['pred_logits'][b, q_idx]
                    target_label = torch.zeros(self.num_classes, device=device)
                    target_label[cls_id] = 1.0
                    
                    ce_loss = F.binary_cross_entropy_with_logits(
                        pred_logit, target_label
                    )
                    batch_loss += ce_loss * self.weight_dict['loss_ce']
                    
                    # Mask loss
                    pred_mask = outputs['pred_masks'][b, q_idx]
                    
                    if pred_mask.shape != damage_mask.shape:
                        pred_mask = F.interpolate(
                            pred_mask.unsqueeze(0).unsqueeze(0),
                            size=damage_mask.shape,
                            mode='bilinear'
                        ).squeeze()
                    
                    mask_loss = F.binary_cross_entropy_with_logits(
                        pred_mask.flatten(),
                        damage_mask.float().flatten()
                    )
                    batch_loss += mask_loss * self.weight_dict['loss_mask']
                    
                    # Dice loss
                    pred_mask_sigmoid = pred_mask.sigmoid().flatten()
                    target_mask_flat = damage_mask.float().flatten()
                    
                    numerator = 2 * (pred_mask_sigmoid * target_mask_flat).sum()
                    denominator = pred_mask_sigmoid.sum() + target_mask_flat.sum()
                    dice_loss = 1 - (numerator + 1) / (denominator + 1)
                    
                    batch_loss += dice_loss * self.weight_dict['loss_dice']
            
            # 정규화
            num_selected = len(active_classes) * self.samples_per_class
            if num_selected > 0:
                total_loss += batch_loss / num_selected
        
        losses['total_loss'] = total_loss / batch_size if batch_size > 0 else total_loss
        
        return total_loss, losses


# 메모리 프로파일링 함수
def profile_memory_usage(num_queries_list=[100, 200, 300, 400]):
    """다양한 query 수에 대한 메모리 사용량 테스트"""
    
    import torch.cuda as cuda
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    for num_q in num_queries_list:
        queries_per_class = num_q // 3
        
        model = Mask2FormerIncreasedQueries(
            queries_per_class=queries_per_class
        ).to(device)
        
        # Dummy input
        features = {
            'P4': torch.randn(2, 256, 40, 40, device=device)
        }
        
        if device == 'cuda':
            cuda.empty_cache()
            cuda.reset_peak_memory_stats()
        
        # Forward pass
        outputs = model(features)
        
        if device == 'cuda':
            peak_memory = cuda.max_memory_allocated() / 1024**3  # GB
            print(f"Queries: {num_q}, Memory: {peak_memory:.2f} GB")
        
        del model, outputs
        
        if device == 'cuda':
            cuda.empty_cache()


# 사용 예시
if __name__ == "__main__":
    # 모델 생성 (300 queries)
    model = Mask2FormerIncreasedQueries(
        queries_per_class=100,  # 각 클래스당 100개
        dec_layers=3,
        hidden_dim=256
    )
    
    print(f"Total queries: {model.num_queries}")
    print(f"Queries per class: {model.queries_per_class}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 메모리 프로파일
    profile_memory_usage([100, 200, 300, 400])