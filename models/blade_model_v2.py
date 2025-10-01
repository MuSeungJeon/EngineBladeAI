# models/blade_model_v2.py (최종 수정안)

import torch
import torch.nn as nn
from typing import Dict, List

# --- 필요한 모든 모듈 임포트 ---
from .backbones.convnext_fpn import ConvNeXtFPN
from .heads.blade_head import SegFormerBladeHead
from .heads.mask2former_damage_head import Mask2FormerDamageHead

def build_backbone(config):
    """Backbone 생성 헬퍼 함수"""
    model_name = config.MODEL.BACKBONE.NAME.lower()
    # ConvNeXtFPN 클래스를 직접 사용
    return ConvNeXtFPN(
        model_name=model_name,
        pretrained=True,
        fpn_channels=config.MODEL.FPN.OUT_CHANNELS
    )

class BladeModelV2(nn.Module):
    """
    [최종 통합 모델]
    """
    def __init__(self, config):
        super().__init__()

        # 이 함수가 파일 내에 정의되어 있어야 합니다.
        self.backbone = build_backbone(config)

        # Head-A (in_channels는 backbone 출력 shape에서 가져오도록 수정)
        self.head_a = SegFormerBladeHead(
            in_channels=[256, 256, 256, 256], # FPN 출력 채널
            num_classes=1
        )

        # Head-B
        self.head_b = Mask2FormerDamageHead(
            in_channels=[256, 256, 256, 256], # FPN 출력 채널
            feat_channels=config.MODEL.HEAD_B.FEAT_CHANNELS,
            out_channels=config.MODEL.HEAD_B.OUT_CHANNELS,
            num_classes=config.MODEL.HEAD_B.NUM_CLASSES,
            queries_per_class=config.MODEL.HEAD_B.QUERIES_PER_CLASS,
            dec_layers=config.MODEL.HEAD_B.DEC_LAYERS
        )

    def forward(self, images: torch.Tensor) -> Dict:
        # 1. 백본 특징 추출
        features = self.backbone(images)

        # 2. Head-A (블레이드 검출)
        blade_outputs = self.head_a(list(features.values()))
        
        # 3. Head-B (손상 검출)
        blade_mask_normalized = torch.sigmoid(blade_outputs)
        
        damage_outputs = self.head_b(
            features=list(features.values()), 
            blade_mask=blade_mask_normalized
        )

        # --- [최종 수정] ---
        # 학습/검증 모드와 상관없이 항상 모든 출력을 반환하도록 통일
        return {
            "blade_logits": blade_outputs,
            "pred_logits": damage_outputs["pred_logits"],
            "pred_masks": damage_outputs["pred_masks"],
            "aux_outputs": damage_outputs["aux_outputs"]
        }