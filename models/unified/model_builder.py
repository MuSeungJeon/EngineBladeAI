# models/unified/model_builder.py
"""모델 빌더: 설정 기반으로 모델 자동 생성"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any

from models.backbones.convnext_fpn import ConvNeXtFPN
from models.heads.blade_head import SegFormerBladeHead
from models.heads.damage_head import DamageDetectionHead


class ModelBuilder:
    """모델 생성 및 관리 클래스"""
    
    # 기본 설정
    DEFAULT_CONFIG = {
        'backbone': {
            'type': 'convnext_fpn',
            'model_name': 'tiny',
            'pretrained': True,
            'fpn_channels': 256,
            'use_fpn': True
        },
        'blade_head': {
            'in_channels': 256,
            'num_classes': 2,
            'dropout_rate': 0.1
        },
        'damage_head': {
            'in_channels': 256,
            'num_classes': 3,
            'num_queries': 200,
            'hidden_dim': 256,
            'num_heads': 8,
            'dec_layers': 6
        },
        'training': {
            'freeze_backbone': False,
            'freeze_blade_head': True,
            'blade_checkpoint': None,
            'damage_checkpoint': None
        }
    }
    
    @classmethod
    def build_backbone(cls, config: Dict = None) -> nn.Module:
        """백본 생성"""
        cfg = cls.DEFAULT_CONFIG['backbone'].copy()
        if config:
            cfg.update(config)
        
        if cfg['type'] == 'convnext_fpn':
            return ConvNeXtFPN(
                model_name=cfg['model_name'],
                pretrained=cfg['pretrained'],
                fpn_channels=cfg['fpn_channels'],
                use_fpn=cfg['use_fpn']
            )
        else:
            raise ValueError(f"Unknown backbone type: {cfg['type']}")
    
    @classmethod
    def build_blade_head(cls, config: Dict = None) -> nn.Module:
        """블레이드 헤드 생성"""
        cfg = cls.DEFAULT_CONFIG['blade_head'].copy()
        if config:
            cfg.update(config)
        
        return SegFormerBladeHead(
            in_channels=cfg['in_channels'],
            num_classes=cfg['num_classes'],
            dropout_rate=cfg['dropout_rate']
        )
    
    @classmethod
    def build_damage_head(cls, config: Dict = None) -> nn.Module:
        cfg = cls.DEFAULT_CONFIG['damage_head'].copy()
        if config:
            cfg.update(config)
        
        # Mask2Former로 변경
        from models.heads.mask2former_damage_head import Mask2FormerDamageHead
        return Mask2FormerDamageHead(
            in_channels=cfg['in_channels'],
            num_classes=cfg['num_classes'],
            num_queries=cfg.get('num_queries', 200),
            hidden_dim=cfg.get('hidden_dim', 256),
            num_heads=cfg.get('num_heads', 8),
            dec_layers=cfg.get('dec_layers', 6)
        )
    
    @classmethod
    def build_unified_model(cls, config: Dict = None) -> nn.Module:
        """통합 모델 생성"""
        from models.unified.unified_model import UnifiedModel
        
        # 전체 설정 병합
        cfg = cls.DEFAULT_CONFIG.copy()
        if config:
            for key in config:
                if key in cfg:
                    cfg[key].update(config[key])
                else:
                    cfg[key] = config[key]
        
        # UnifiedModel이 받는 실제 파라미터로 생성
        model = UnifiedModel(
            backbone_type=cfg['backbone'].get('model_name', 'tiny'),
            num_blade_classes=cfg['blade_head'].get('num_classes', 2),
            num_damage_classes=cfg['damage_head'].get('num_classes', 3),
            pretrained_backbone=cfg['backbone'].get('pretrained', True),
            blade_checkpoint=cfg['training'].get('blade_checkpoint'),
            freeze_blade=cfg['training'].get('freeze_blade_head', True),
            use_fpn=cfg['backbone'].get('use_fpn', True)
        )
        
        # Freeze 설정은 모델 내부에서 이미 처리됨
        if cfg['training'].get('freeze_backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("✅ Backbone frozen")
        
        return model
    
    @staticmethod
    def load_checkpoint(
        module: nn.Module,
        checkpoint_path: str,
        prefix: str = None,
        strict: bool = False
    ):
        """체크포인트 로드"""
        if not Path(checkpoint_path).exists():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Prefix 처리
        if prefix:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    new_key = k.replace(f"{prefix}.", "")
                    new_state_dict[new_key] = v
            state_dict = new_state_dict
        
        module.load_state_dict(state_dict, strict=strict)
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> nn.Module:
        """YAML 설정 파일로부터 모델 생성"""
        import yaml
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls.build_unified_model(config)
    
    @classmethod
    def create_model_zoo(cls) -> Dict[str, nn.Module]:
        """사전 정의된 모델들 생성"""
        
        model_zoo = {}
        
        # 1. 가장 가벼운 모델
        model_zoo['tiny_fast'] = cls.build_unified_model({
            'backbone': {'model_name': 'tiny', 'use_fpn': False},
            'blade_head': {'dropout_rate': 0.0},
            'damage_head': {'num_classes': 3}
        })
        
        # 2. 균형잡힌 모델 (권장)
        model_zoo['balanced'] = cls.build_unified_model({
            'backbone': {'model_name': 'tiny', 'use_fpn': True},
            'blade_head': {'dropout_rate': 0.1},
            'damage_head': {'num_classes': 3}
        })
        
        # 3. 최고 성능 모델
        model_zoo['best'] = cls.build_unified_model({
            'backbone': {'model_name': 'small', 'use_fpn': True},
            'blade_head': {'dropout_rate': 0.1},
            'damage_head': {'num_classes': 3}
        })
        
        return model_zoo


class ModelConfigValidator:
    """모델 설정 검증"""
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """설정 유효성 검사"""
        required_keys = ['backbone', 'blade_head', 'damage_head']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        # 백본 검증
        if config['backbone'].get('use_fpn'):
            fpn_channels = config['backbone'].get('fpn_channels', 256)
            blade_channels = config['blade_head'].get('in_channels', 256)
            damage_channels = config['damage_head'].get('in_channels', 256)
            
            if not (fpn_channels == blade_channels == damage_channels):
                raise ValueError(
                    f"Channel mismatch: FPN={fpn_channels}, "
                    f"Blade={blade_channels}, Damage={damage_channels}"
                )
        
        return True