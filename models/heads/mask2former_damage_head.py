# models/heads/mask2former_damage_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import torch.utils.checkpoint as checkpoint

class Mask2FormerDamageHead(nn.Module):
    """
    [최종 통합 버전]
    - ImprovedPixelDecoder로 고품질 마스크 특징 생성 (Version A)
    - Gaussian Constraint로 블레이드 영역 정보 활용 (Version A)
    - 클래스별 특화 쿼리 (300개) 및 Class Mask 적용 (Version B)
    - Gradient Checkpointing으로 메모리 최적화 (Version B)
    """

    def __init__(
        self,
        in_channels: List[int] = [256, 256, 256, 256],
        feat_channels: int = 256,
        out_channels: int = 256,
        num_classes: int = 3,
        queries_per_class: int = 100, # --- [통합] 클래스당 쿼리 수 ---
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dec_layers: int = 6, # 성능을 위해 레이어 수 증가 고려
        dropout: float = 0.1,
        use_blade_mask: bool = True,
        enforce_class_specialization: bool = True # --- [통합] 클래스 특화 강제 ---
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_blade_mask = use_blade_mask
        self.enforce_class_specialization = enforce_class_specialization

        # --- [통합] 클래스 특화 쿼리 설정 ---
        self.queries_per_class = queries_per_class
        self.num_queries = self.num_classes * self.queries_per_class

        # Gaussian constraint parameters
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.min_weight = nn.Parameter(torch.tensor(0.1))
        self.blade_weight = nn.Parameter(torch.tensor(0.9))

        # --- [Version A] 고품질 마스크 생성을 위한 Pixel Decoder ---
        self.pixel_decoder = ImprovedPixelDecoder(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
            num_levels=len(in_channels)
        )

        # Transformer Decoder
        self.transformer_decoder = TransformerDecoder(
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            return_intermediate=True
        )

        # Query embeddings
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # Output heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.mask_head = MaskHead(hidden_dim, hidden_dim, mask_dim=out_channels)

        # --- [통합] 클래스 특화 마스크 생성 ---
        if self.enforce_class_specialization:
            self.register_buffer('class_mask', self._create_class_mask())

    def _create_class_mask(self):
        """각 query가 전문 클래스만 높게 예측하도록 하는 마스크 생성"""
        mask = torch.ones(self.num_queries, self.num_classes) * -10.0 # 기본적으로 낮은 값으로 억제
        for q_idx in range(self.num_queries):
            class_idx = q_idx // self.queries_per_class
            mask[q_idx, class_idx] = 0.0 # 자신의 클래스에는 영향 없음 (0을 더함)
        return mask

    def apply_gaussian_constraint(self, features, blade_mask):
        """Gaussian distance-based soft constraint 적용"""
        from kornia.contrib import distance_transform

        # blade_mask shape: [B, 1, H, W]
        
        with torch.no_grad():
            # --- [최종 수정] 4D 텐서를 그대로 전달 ---
            # distance_transform 함수는 BxCxHxW 형태를 기대합니다.
            distance = distance_transform((1 - blade_mask).float())

        # distance도 [B, 1, H, W] 이므로 squeeze로 차원 축소
        distance = distance.squeeze(1)

        sigma_safe = torch.abs(self.sigma) + 1e-6
        decay = torch.exp(-distance**2 / (2 * sigma_safe**2))
        
        min_w = torch.sigmoid(self.min_weight)
        blade_w = torch.sigmoid(self.blade_weight)
        
        decay_clamped = torch.max(decay, min_w)
        
        constraint_mask = (blade_mask * blade_w) + ((1 - blade_mask) * decay_clamped.unsqueeze(1))
        
        constrained_features = []
        for feat in features:
            mask_resized = F.interpolate(
                constraint_mask,
                size=feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            constrained_features.append(feat * mask_resized)
        return constrained_features

    def forward(
        self,
        features: List[torch.Tensor],
        blade_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        batch_size = features[0].size(0)

        if blade_mask is not None and self.use_blade_mask:
            features = self.apply_gaussian_constraint(features, blade_mask)

        mask_features, multi_scale_features = self.pixel_decoder(features)

        memory = []
        for feat in multi_scale_features:
            memory.append(feat.flatten(2).transpose(1, 2))
        memory = torch.cat(memory, dim=1)

        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        if self.training:
            decoder_outputs = checkpoint.checkpoint(
                self.transformer_decoder, query_embeds, memory, use_reentrant=False
            )
        else:
            decoder_outputs = self.transformer_decoder(query_embeds, memory)

        # --- [최종 수정] decoder_outputs에서 마지막 레이어 출력만 사용 ---
        # decoder_outputs shape: [layers, batch, queries, channels]
        # outputs_class와 mask_embed 계산 시 마지막 레이어 출력만 사용
        outputs_class = self.class_head(decoder_outputs)
        mask_embed = self.mask_head(decoder_outputs)
        
        # einsum을 위한 mask_embed는 마지막 레이어의 결과만 사용: [batch, queries, channels]
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed[-1], mask_features)
        
        if self.enforce_class_specialization:
            # 클래스 특화는 모든 레이어 출력에 적용
            outputs_class = outputs_class + self.class_mask.unsqueeze(0).unsqueeze(0)

        return {
            'pred_logits': outputs_class[-1], # 최종 출력도 마지막 레이어 결과
            'pred_masks': outputs_mask,       # 마스크는 이미 마지막 레이어로 계산됨
            'aux_outputs': self._set_aux_loss(outputs_class[:-1], mask_embed[:-1], mask_features)
        }

    def _set_aux_loss(self, outputs_class, mask_embeds, mask_features):
        # 보조 손실(aux_loss) 계산을 위해 각 레이어의 mask_embed와 mask_features를 곱해줌
        outputs_masks = []
        for mask_embed in mask_embeds:
            # einsum 연산을 통해 각 보조 레이어의 마스크 예측 생성
            outputs_masks.append(torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features))
            
        return [
            {'pred_logits': c, 'pred_masks': m}
            for c, m in zip(outputs_class, outputs_masks)
        ]
    
    

# --------------------------------------------------------------------------------
# 아래는 Mask2FormerDamageHead가 사용하는 서브 모듈들입니다.
# --------------------------------------------------------------------------------

class ImprovedPixelDecoder(nn.Module):
    """FPN 기반의 Pixel Decoder"""
    def __init__(self, in_channels, feat_channels, out_channels, num_levels):
        super().__init__()
        self.num_levels = num_levels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, feat_channels, 1) for ch in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1) for _ in range(num_levels)
        ])
        self.fusion_conv = nn.Conv2d(feat_channels * num_levels, feat_channels, 1)
        self.output_conv = nn.Conv2d(feat_channels, out_channels, 3, padding=1)

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(self.num_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='bilinear', align_corners=False)

        fpn_features = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        
        # 최상위 레벨(가장 작은) 특징 맵 크기로 모두 리사이즈하여 융합
        target_size = fpn_features[-1].shape[-2:]
        fused_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in fpn_features]
        fused = self.fusion_conv(torch.cat(fused_features, dim=1))
        
        mask_features = self.output_conv(fused)
        return mask_features, fpn_features

class MaskHead(nn.Module):
    """Mask prediction head"""
    def __init__(self, in_dim, hidden_dim, mask_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, mask_dim)
        )
    def forward(self, x):
        return self.layers(x)

class TransformerDecoder(nn.Module):
    """Standard Transformer Decoder"""
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward, dropout, return_intermediate):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, activation='relu', batch_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, **kwargs):
        intermediate = []
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, **kwargs)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        return self.norm(output).unsqueeze(0)