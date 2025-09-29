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
    """
    Mask2Former 기반 손상 검출 헤드 - Gaussian constraint 포함
    300 queries = 100 per class (crack, nick, tear)
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        queries_per_class: int = 100,  # 클래스당 쿼리 수
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dec_layers: int = 3,
        dropout: float = 0.1,
        use_blade_mask: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.num_queries = num_classes * queries_per_class  # 300 = 3 × 100
        self.use_blade_mask = use_blade_mask
        
        # Gaussian constraint parameters (Head-B 소유)
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.min_weight = nn.Parameter(torch.tensor(0.1))
        self.blade_weight = nn.Parameter(torch.tensor(0.9))
        
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
        
        # Query embeddings - 클래스별로 초기화
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.level_embed = nn.Embedding(3, hidden_dim)
        
        # Class-specific query initialization
        self._init_class_queries()
        
        # Output heads
        self.class_head = nn.Linear(hidden_dim, num_classes) 
        self.mask_head = MaskHead(hidden_dim, hidden_dim, mask_dim=hidden_dim)
        
        # Multi-label classification (global prediction)
        self.multilabel_head = nn.Sequential(
            nn.Linear(hidden_dim * self.num_queries, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Class-specific attention weights (optional)
        self.class_attention_weights = nn.Parameter(
            torch.ones(num_classes, queries_per_class) / queries_per_class
        )
        
        # Blade mask attention
        if use_blade_mask:
            self.blade_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
    
    def _init_class_queries(self):
        """
        Initialize queries by damage type for better convergence
        Each class gets dedicated queries:
        - Queries 0-99: Crack
        - Queries 100-199: Nick  
        - Queries 200-299: Tear
        """
        nn.init.normal_(self.query_embed.weight, std=0.01)
        
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                # Each class gets its own query range
                start_idx = class_idx * self.queries_per_class
                end_idx = (class_idx + 1) * self.queries_per_class
                
                # Add class-specific bias to help queries specialize
                # Different initialization per class
                if class_idx == 0:  # Crack - often linear
                    self.query_embed.weight[start_idx:end_idx] *= 1.0
                elif class_idx == 1:  # Nick - often small
                    self.query_embed.weight[start_idx:end_idx] *= 0.8
                elif class_idx == 2:  # Tear - often irregular
                    self.query_embed.weight[start_idx:end_idx] *= 1.2
                
                # Add small class-specific offset
                self.query_embed.weight[start_idx:end_idx] += class_idx * 0.01
    
    def apply_gaussian_constraint(self, features, blade_mask):
        """Apply Gaussian distance-based soft constraint"""
        
        # Use scipy for distance transform (more stable than kornia)
        from scipy.ndimage import distance_transform_edt
        import numpy as np
        
        batch_size = blade_mask.shape[0]
        constraint_masks = []
        
        for b in range(batch_size):
            # Convert to numpy for distance transform
            blade_np = blade_mask[b].cpu().numpy()
            
            # Compute distance from blade edge
            with torch.no_grad():
                # Distance from non-blade regions
                distance = distance_transform_edt(1 - blade_np)
                distance = torch.from_numpy(distance).to(blade_mask.device).float()
            
            # Gaussian decay
            sigma = torch.abs(self.sigma) + 1e-6
            decay = torch.exp(-distance**2 / (2 * sigma**2))
            
            # Create constraint mask
            min_w = torch.sigmoid(self.min_weight)
            blade_w = torch.sigmoid(self.blade_weight)
            
            constraint_mask = blade_mask[b] * blade_w + (1 - blade_mask[b]) * torch.clamp(decay, min=min_w)
            constraint_masks.append(constraint_mask)
        
        constraint_mask = torch.stack(constraint_masks)
        
        # Apply to features
        constrained_features = []
        for feat in features:
            h, w = feat.shape[-2:]
            mask_resized = F.interpolate(
                constraint_mask.unsqueeze(1),
                size=(h, w),
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
        
        # Apply Gaussian constraint if blade mask provided
        if blade_mask is not None and self.use_blade_mask:
            features = self.apply_gaussian_constraint(features, blade_mask)
        
        # 1. Pixel Decoder: multi-scale features → mask features
        mask_features, multi_scale_features = self.pixel_decoder(features)
        
        # 2. Prepare queries (300 total)
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
        
        # 5. Generate outputs with class-specific processing
        outputs_class = []
        outputs_mask = []
        
        for layer_output in decoder_outputs:
            # Class prediction for all queries
            class_pred = self.class_head(layer_output)
            
            # Apply class-specific biases to predictions
            # Each query range gets a bias toward its designated class
            for class_idx in range(self.num_classes):
                start_idx = class_idx * self.queries_per_class
                end_idx = (class_idx + 1) * self.queries_per_class
                
                # Add soft bias to encourage specialization
                bias = torch.zeros_like(class_pred[:, start_idx:end_idx])
                bias[:, :, class_idx] = 1.0  # Bias toward designated class
                class_pred[:, start_idx:end_idx] = class_pred[:, start_idx:end_idx] + bias * 0.1
            
            outputs_class.append(class_pred)
            
            # Mask prediction
            mask_embed = self.mask_head(layer_output)
            mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            outputs_mask.append(mask_pred)
        
        # 6. Multi-label prediction (using all queries)
        all_queries = decoder_outputs[-1].flatten(1)
        multilabel = torch.sigmoid(self.multilabel_head(all_queries))
        
        # 7. Optional: Class-weighted aggregation for better multi-label prediction
        # Aggregate predictions from each query group
        class_aggregated = []
        final_output = decoder_outputs[-1]  # [B, 300, 256]
        
        for class_idx in range(self.num_classes):
            start_idx = class_idx * self.queries_per_class
            end_idx = (class_idx + 1) * self.queries_per_class
            
            # Get queries for this class
            class_queries = final_output[:, start_idx:end_idx]  # [B, 100, 256]
            
            # Weighted average using attention weights
            weights = self.class_attention_weights[class_idx].softmax(dim=0)
            weighted_query = torch.einsum('bqd,q->bd', class_queries, weights)
            class_aggregated.append(weighted_query)
        
        # Stack and create alternative multi-label prediction
        class_features = torch.stack(class_aggregated, dim=1)  # [B, 3, 256]
        multilabel_alt = torch.sigmoid(
            class_features.mean(dim=2)  # Simple average per class
        )
        
        return {
            'pred_logits': outputs_class[-1],  # [B, 300, 3]
            'pred_masks': outputs_mask[-1],    # [B, 300, H, W]
            'multilabel': multilabel,           # [B, 3] - from all queries
            'multilabel_alt': multilabel_alt,   # [B, 3] - from class-specific aggregation
            'aux_outputs': [
                {'pred_logits': c, 'pred_masks': m}
                for c, m in zip(outputs_class[:-1], outputs_mask[:-1])
            ],
            # Additional info for analysis
            'queries_per_class': {
                'crack': (0, self.queries_per_class),
                'nick': (self.queries_per_class, 2 * self.queries_per_class),
                'tear': (2 * self.queries_per_class, 3 * self.queries_per_class)
            }
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