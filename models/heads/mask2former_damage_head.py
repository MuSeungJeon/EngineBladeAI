# models/heads/mask2former_damage_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math


class Mask2FormerDamageHead(nn.Module):
    """Mask2Former 기반 손상 검출 헤드 - 작은 결함 최적화"""
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        num_queries: int = 200,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dec_layers: int = 6,
        dropout: float = 0.1,
        use_blade_mask: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.use_blade_mask = use_blade_mask
        
        # Pixel Decoder (FPN features → pixel features)
        self.pixel_decoder = MSDeformAttnPixelDecoder(
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
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(3, hidden_dim)
        
        # Output heads
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
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
                dropout=dropout
            )
    
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
        # query_pos, memory_pos 제거
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
                    blade_mask.float(),
                    size=(h, w),
                    mode='nearest'
                )
                mask_flat = (mask_resized < 0.5).flatten(1)
                memory_mask_list.append(mask_flat)
            else:
                memory_mask_list.append(torch.zeros((bs, h*w), device=feat.device, dtype=torch.bool))
        
        memory = torch.cat(memory_list, dim=1)
        memory_mask = torch.cat(memory_mask_list, dim=1) if blade_mask is not None else None
        
        return memory, memory_mask
    
    def get_positional_encoding(self, memory):
        """Generate positional encoding for memory"""
        bs, hw, c = memory.shape
        pos_encoding = torch.zeros_like(memory)
        
        position = torch.arange(hw, device=memory.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, c, 2, device=memory.device) * 
                            -(math.log(10000.0) / c))
        
        pos_encoding[:, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, :, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def apply_blade_constraint(self, masks, blade_mask):
        """Apply blade mask constraint with soft boundaries"""
        kernel_size = 21
        padding = kernel_size // 2
        
        blade_mask_expanded = F.max_pool2d(
            blade_mask.float(),
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        constrained_masks = []
        for mask_batch in masks:
            mask_resized = F.interpolate(
                blade_mask_expanded,
                size=mask_batch.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            soft_mask = 0.3 + 0.7 * mask_resized
            constrained = mask_batch * soft_mask.unsqueeze(1)
            constrained_masks.append(constrained)
        
        return constrained_masks


class MSDeformAttnPixelDecoder(nn.Module):
    """Multi-Scale Deformable Attention Pixel Decoder"""
    
    def __init__(
        self,
        in_channels: List[int],
        feat_channels: int,
        out_channels: int,
        num_levels: int = 3
    ):
        super().__init__()
        
        self.num_levels = num_levels
        
        self.input_projects = nn.ModuleList()
        for in_ch in in_channels:
            self.input_projects.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feat_channels, 1),
                    nn.GroupNorm(32, feat_channels)
                )
            )
        
        self.encoder = nn.ModuleList([
            DeformableAttentionBlock(
                embed_dim=feat_channels,
                num_heads=8,
                num_levels=num_levels,
                num_points=4
            )
            for _ in range(2)
        ])
        
        self.output_project = nn.Sequential(
            nn.Conv2d(feat_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU()
        )
    
    def forward(self, features):
        projected = []
        for feat, proj in zip(features, self.input_projects):
            projected.append(proj(feat))
        
        multi_scale_features = projected[-self.num_levels:]
        
        encoded = multi_scale_features[-1]
        for encoder_layer in self.encoder:
            encoded = encoder_layer(encoded, multi_scale_features)
        
        mask_features = self.output_project(encoded)
        
        target_size = features[0].shape[-2:]
        mask_features = F.interpolate(
            mask_features,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        return mask_features, multi_scale_features


class DeformableAttentionBlock(nn.Module):
    """Simplified Deformable Attention"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_points = num_points
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        self.sampling_offsets = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points * 2
        )
        
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, query, reference_features):
        bs, c, h, w = query.shape
        query_flat = query.flatten(2).permute(2, 0, 1)
        
        refs = []
        for ref in reference_features:
            ref_resized = F.interpolate(ref, size=(h, w), mode='bilinear', align_corners=False)
            refs.append(ref_resized.flatten(2).permute(2, 0, 1))
        
        memory = torch.cat(refs, dim=0)
        
        attn_output, _ = self.attention(query_flat, memory, memory)
        
        output = attn_output.permute(1, 2, 0).reshape(bs, c, h, w)
        
        output = self.norm(query.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(bs, c, h, w) + output
        
        return output


class MaskHead(nn.Module):
    """Mask prediction head"""
    
    def __init__(self, in_dim, hidden_dim, mask_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
        num_layers=6,
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


class Mask2FormerLoss(nn.Module):
    """Mask2Former Loss with Hungarian Matching"""
    
    def __init__(self, num_classes=3, aux_loss=True, aux_weight=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.aux_weight = aux_weight
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = SigmoidFocalLoss()
        
        # Loss weights
        self.class_weight = 2.0
        self.mask_weight = 5.0
        self.dice_weight = 5.0
        
        # Hungarian Matcher
        self.matcher = HungarianMatcher(
            cost_class=self.class_weight,
            cost_mask=self.mask_weight,
            cost_dice=self.dice_weight
        )
    
    def forward(self, outputs, targets):
        """
        outputs: dict with 'pred_logits', 'pred_masks', 'multilabel', 'aux_outputs'
        targets: dict with 'instance_masks', 'instance_labels', 'multilabel'
        """
        losses = {}
        
        # Multi-label loss (always computed)
        if 'multilabel' in outputs and 'multilabel' in targets:
            losses['multilabel'] = self.bce_loss(outputs['multilabel'], targets['multilabel'])
        
        # Instance matching and losses
        if 'pred_logits' in outputs and 'pred_masks' in outputs:
            # Get matched indices
            indices = self.matcher(outputs, targets)
            
            # Classification loss
            losses['ce'] = self.get_loss_ce(outputs, targets, indices)
            
            # Mask losses
            losses['mask'] = self.get_loss_mask(outputs, targets, indices)
            losses['dice'] = self.get_loss_dice(outputs, targets, indices)
        
        # Auxiliary losses
        if self.aux_loss and 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'pred_logits' in aux_outputs and 'pred_masks' in aux_outputs:
                    aux_indices = self.matcher(aux_outputs, targets)
                    
                    aux_ce = self.get_loss_ce(aux_outputs, targets, aux_indices)
                    aux_mask = self.get_loss_mask(aux_outputs, targets, aux_indices)
                    aux_dice = self.get_loss_dice(aux_outputs, targets, aux_indices)
                    
                    losses[f'ce_{i}'] = aux_ce * self.aux_weight
                    losses[f'mask_{i}'] = aux_mask * self.aux_weight
                    losses[f'dice_{i}'] = aux_dice * self.aux_weight
        
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss, losses
    
    def get_loss_ce(self, outputs, targets, indices):
        """Classification loss"""
        pred_logits = outputs['pred_logits']
        
        # Prepare targets
        target_classes = torch.full(
            pred_logits.shape[:2], 
            self.num_classes,  # background class
            dtype=torch.long, 
            device=pred_logits.device
        )
        
        # Set matched queries to correct classes
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0 and 'instance_labels' in targets:
                if targets['instance_labels'][batch_idx] is not None:
                    target_classes[batch_idx, pred_idx] = targets['instance_labels'][batch_idx][tgt_idx]
        
        loss = F.cross_entropy(
            pred_logits.flatten(0, 1),
            target_classes.flatten(),
            reduction='mean'
        )
        
        return loss * self.class_weight
    
    def get_loss_mask(self, outputs, targets, indices):
        """Mask loss (Focal loss)"""
        pred_masks = outputs['pred_masks']
        
        total_loss = 0
        num_masks = 0
        
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0 and 'instance_masks' in targets:
                if targets['instance_masks'][batch_idx] is not None:
                    pred = pred_masks[batch_idx, pred_idx]
                    target = targets['instance_masks'][batch_idx][tgt_idx]
                    
                    # Resize target if needed
                    if pred.shape[-2:] != target.shape[-2:]:
                        target = F.interpolate(
                            target.unsqueeze(1).float(),
                            size=pred.shape[-2:],
                            mode='nearest'
                        ).squeeze(1)
                    
                    loss = self.focal_loss(pred.flatten(1), target.flatten(1))
                    total_loss += loss
                    num_masks += len(pred_idx)
        
        if num_masks > 0:
            return (total_loss / num_masks) * self.mask_weight
        return total_loss * self.mask_weight
    
    def get_loss_dice(self, outputs, targets, indices):
        """Dice loss"""
        pred_masks = outputs['pred_masks']
        
        total_loss = 0
        num_masks = 0
        
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0 and 'instance_masks' in targets:
                if targets['instance_masks'][batch_idx] is not None:
                    pred = pred_masks[batch_idx, pred_idx].sigmoid()
                    target = targets['instance_masks'][batch_idx][tgt_idx]
                    
                    # Resize target if needed
                    if pred.shape[-2:] != target.shape[-2:]:
                        target = F.interpolate(
                            target.unsqueeze(1).float(),
                            size=pred.shape[-2:],
                            mode='nearest'
                        ).squeeze(1)
                    
                    loss = self.dice_loss(pred, target)
                    total_loss += loss
                    num_masks += len(pred_idx)
        
        if num_masks > 0:
            return (total_loss / num_masks) * self.dice_weight
        return total_loss * self.dice_weight


class DiceLoss(nn.Module):
    """Dice Loss"""
    
    def forward(self, pred, target):
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2 * intersection + 1) / (union + 1)
        return (1 - dice).mean()


class SigmoidFocalLoss(nn.Module):
    """Sigmoid Focal Loss"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        target = target.float()
        
        # Focal weight
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - pt).pow(self.gamma)
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Apply focal weight
        loss = focal_weight * loss
        
        # Apply alpha weight
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * loss
        
        return loss.mean()


class HungarianMatcher(nn.Module):
    """Hungarian Matcher for bipartite matching"""
    
    def __init__(self, cost_class=1.0, cost_mask=1.0, cost_dice=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        from scipy.optimize import linear_sum_assignment
        
        batch_size = outputs['pred_logits'].shape[0]
        indices = []
        
        for b in range(batch_size):
            # 타겟이 없으면 빈 인덱스 반환
            if targets['instance_labels'][b] is None or len(targets['instance_labels'][b]) == 0:
                indices.append((torch.tensor([], dtype=torch.long), 
                            torch.tensor([], dtype=torch.long)))
                continue
            
            pred_masks = outputs['pred_masks'][b]  # [num_queries, H, W]
            tgt_masks = targets['instance_masks'][b]  # [N, H', W']
            
            # 크기 맞추기
            if pred_masks.shape[-2:] != tgt_masks.shape[-2:]:
                tgt_masks = F.interpolate(
                    tgt_masks.unsqueeze(0),
                    size=pred_masks.shape[-2:],
                    mode='nearest'
                ).squeeze(0)
            
            # Flatten
            pred_masks_flat = pred_masks.sigmoid().flatten(1)  # [num_queries, H*W]
            tgt_masks_flat = tgt_masks.flatten(1)  # [N, H*W]
            
            # Cost 계산
            cost_mask = -torch.matmul(pred_masks_flat, tgt_masks_flat.T)
            
            # 간단한 매칭
            C = cost_mask.cpu().numpy()
            pred_idx, tgt_idx = linear_sum_assignment(C)
            
            indices.append((torch.tensor(pred_idx), torch.tensor(tgt_idx)))
        
        return indices
    

    