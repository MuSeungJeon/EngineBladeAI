# models/heads/mask2former_damage_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math

class ImprovedPixelDecoder(nn.Module):
    """개선된 Pixel Decoder - Multi-scale fusion with learnable upsampling"""
    
    def __init__(self, in_channels, feat_channels, out_channels, num_levels):
        super().__init__()
        self.num_levels = num_levels
        
        # Multi-level lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(num_levels):
            self.lateral_convs.append(
                nn.Conv2d(in_channels[i], feat_channels, 1)
            )
        
        # Top-down pathway
        self.fpn_convs = nn.ModuleList()
        for i in range(num_levels):
            self.fpn_convs.append(
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
            )
        
        # Multi-scale fusion
        self.fusion_conv = nn.Conv2d(feat_channels * num_levels, feat_channels, 3, padding=1)
        
        # Learnable upsampling (better than bilinear)
        self.upsample_conv = nn.ConvTranspose2d(
            feat_channels, out_channels, 
            kernel_size=8, stride=4, padding=2
        )
        
        self.output_norm = nn.GroupNorm(32, out_channels)
        self.output_activation = nn.ReLU(inplace=True)
        
    def forward(self, features):
        # Lateral connections
        laterals = []
        for i in range(self.num_levels):
            laterals.append(self.lateral_convs[i](features[i]))
        
        # Top-down path with lateral connections
        for i in range(self.num_levels - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Apply FPN convolutions
        fpn_features = []
        for i in range(self.num_levels):
            fpn_features.append(self.fpn_convs[i](laterals[i]))
        
        # Multi-scale fusion - resize all to first level size
        target_size = fpn_features[0].shape[-2:]
        fused_features = []
        for i in range(self.num_levels):
            if i > 0:
                resized = F.interpolate(
                    fpn_features[i],
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                fused_features.append(resized)
            else:
                fused_features.append(fpn_features[i])
        
        # Concatenate and fuse
        concat_features = torch.cat(fused_features, dim=1)
        fused = self.fusion_conv(concat_features)
        
        # Upsample to 640x640 (4x from 160x160)
        mask_features = self.upsample_conv(fused)
        mask_features = self.output_norm(mask_features)
        mask_features = self.output_activation(mask_features)
        
        return mask_features, fpn_features[:3]


class Mask2FormerDamageHead(nn.Module):
    """Mask2Former 기반 손상 검출 헤드 - 작은 결함 최적화"""
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        num_queries: int = 100,  # 100으로 줄임
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dec_layers: int = 3,  # 3으로 줄임
        dropout: float = 0.1,
        use_blade_mask: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.use_blade_mask = use_blade_mask
        
        # Improved Pixel Decoder
        self.pixel_decoder = ImprovedPixelDecoder(
            in_channels=[256, 256, 256, 256],
            feat_channels=256,
            out_channels=256,
            num_levels=3
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
        
        # Query embeddings - class-aware initialization
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(3, hidden_dim)
        
        # Initialize queries by damage type
        self._init_queries()
        
        # Output heads
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.mask_head = MaskHead(hidden_dim, hidden_dim, mask_dim=hidden_dim)
        
        # Multi-label classification
        self.multilabel_head = nn.Sequential(
            nn.Linear(hidden_dim * num_queries, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Blade mask attention
        if use_blade_mask:
            self.blade_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
    
    def _init_queries(self):
        """Initialize queries by damage type for better convergence"""
        nn.init.normal_(self.query_embed.weight, std=0.01)
        # Divide queries by damage type
        queries_per_class = self.num_queries // 3
        with torch.no_grad():
            for i in range(3):
                start = i * queries_per_class
                end = (i + 1) * queries_per_class if i < 2 else self.num_queries
                self.query_embed.weight[start:end] *= (1.0 + i * 0.1)
    
    def forward(
        self,
        features: List[torch.Tensor],
        blade_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = features[0].size(0)
        
        # 1. Pixel Decoder: multi-scale features → mask features
        mask_features, multi_scale_features = self.pixel_decoder(features)
        
        # 2. Prepare queries
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. Prepare memory (flattened features)
        memory, memory_mask = self.prepare_memory(
            multi_scale_features, blade_mask
        )
        
        # 4. Transformer decoder
        decoder_outputs = self.transformer_decoder(
            tgt=query_embeds,
            memory=memory,
            memory_key_padding_mask=memory_mask
        )
        
        # 5. Generate outputs
        outputs_class = []
        outputs_mask = []
        
        for layer_output in decoder_outputs:
            # Class prediction
            class_pred = self.class_head(layer_output)
            outputs_class.append(class_pred)
            
            # Mask prediction
            mask_embed = self.mask_head(layer_output)
            mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            outputs_mask.append(mask_pred)
        
        # 6. Multi-label prediction
        all_queries = decoder_outputs[-1].flatten(1)
        multilabel = torch.sigmoid(self.multilabel_head(all_queries))
        
        # 7. Apply blade mask if available
        if blade_mask is not None and self.use_blade_mask:
            outputs_mask = self.apply_blade_constraint(outputs_mask, blade_mask)
        
        return {
            'pred_logits': outputs_class[-1],
            'pred_masks': outputs_mask[-1],
            'multilabel': multilabel,
            'aux_outputs': [
                {'pred_logits': c, 'pred_masks': m}
                for c, m in zip(outputs_class[:-1], outputs_mask[:-1])
            ]
        }
    
    def prepare_memory(self, features, blade_mask=None):
        """Prepare memory for transformer"""
        memory_list = []
        memory_mask_list = []
        
        for lvl, feat in enumerate(features):
            bs, c, h, w = feat.shape
            feat_flat = feat.flatten(2).transpose(1, 2)
            memory_list.append(feat_flat)
            
            if blade_mask is not None:
                mask_resized = F.interpolate(
                    blade_mask.unsqueeze(1).float(),
                    size=(h, w),
                    mode='nearest'
                ).squeeze(1)
                mask_flat = (mask_resized < 0.5).flatten(1)
                memory_mask_list.append(mask_flat)
            else:
                memory_mask_list.append(torch.zeros((bs, h*w), device=feat.device, dtype=torch.bool))
        
        memory = torch.cat(memory_list, dim=1)
        memory_mask = torch.cat(memory_mask_list, dim=1) if blade_mask is not None else None
        
        return memory, memory_mask
    
    def apply_blade_constraint(self, masks, blade_mask):
        """Apply blade mask constraint with soft boundaries"""
        kernel_size = 21
        padding = kernel_size // 2
        
        # Expand blade mask with max pooling
        blade_mask_expanded = F.max_pool2d(
            blade_mask.unsqueeze(1).float(),
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        ).squeeze(1)
        
        constrained_masks = []
        for mask_batch in masks:
            # Resize blade mask to match prediction size
            mask_resized = F.interpolate(
                blade_mask_expanded.unsqueeze(1),
                size=mask_batch.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # Soft mask: 0.3 (background) + 0.7 (blade region)
            soft_mask = 0.3 + 0.7 * mask_resized
            constrained = mask_batch * soft_mask.unsqueeze(1)
            constrained_masks.append(constrained)
        
        return constrained_masks


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
    
    def __init__(
        self,
        d_model=256,
        num_heads=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        return_intermediate=True
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
            for _ in range(num_layers)
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
        
        if not self.return_intermediate:
            return self.norm(output).unsqueeze(0)
        
        return torch.stack(intermediate)