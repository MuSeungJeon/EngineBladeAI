# models/backbones/convnext_fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, convnext_small, convnext_base


class FeaturePyramidNetwork(nn.Module):
    """제대로 된 Feature Pyramid Network"""
    
    def __init__(
        self,
        in_channels_list,
        out_channels=256,
        extra_blocks=None,
        norm_layer=None
    ):
        super().__init__()
        
        # Lateral connections (1x1 conv)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                norm_layer(out_channels) if norm_layer else nn.Identity()
            )
            self.lateral_convs.append(lateral)
        
        # Output convolutions (3x3 conv)
        self.output_convs = nn.ModuleList()
        for _ in in_channels_list:
            output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                norm_layer(out_channels) if norm_layer else nn.Identity(),
                nn.ReLU(inplace=True)
            )
            self.output_convs.append(output_conv)
        
        # Extra blocks (P6, P7)
        self.extra_blocks = extra_blocks
        if extra_blocks == 'pool':
            # MaxPool for P6
            self.p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        elif extra_blocks == 'conv':
            # Strided conv for P6, P7
            self.p6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: list of feature maps from backbone
                [C1, C2, C3, C4] or [C2, C3, C4, C5]
        Returns:
            fpn_features: list of FPN feature maps
                [P2/P3, P3/P4, P4/P5, P5/P6, ...]
        """
        
        # Build lateral features
        laterals = []
        for feat, lateral_conv in zip(features, self.lateral_convs):
            laterals.append(lateral_conv(feat))
        
        # Build top-down path
        # Start from deepest layer
        used_laterals = []
        prev_features = laterals[-1]
        used_laterals.append(prev_features)
        
        # Propagate down (from deep to shallow)
        for lateral_feat in laterals[-2::-1]:  # Reverse, skip last
            # Upsample deeper feature
            top_down = F.interpolate(
                prev_features,
                size=lateral_feat.shape[-2:],
                mode='nearest'
            )
            # Add lateral connection
            prev_features = lateral_feat + top_down
            used_laterals.insert(0, prev_features)
        
        # Apply output convolutions
        fpn_features = []
        for lateral_feat, output_conv in zip(used_laterals, self.output_convs):
            fpn_features.append(output_conv(lateral_feat))
        
        # Add extra levels if specified
        if self.extra_blocks == 'pool':
            # P6 is obtained via a max pool on P5
            p6 = self.p6(fpn_features[-1])
            fpn_features.append(p6)
        elif self.extra_blocks == 'conv':
            # P6 via strided conv on C5/P5
            p6 = self.p6(fpn_features[-1])
            fpn_features.append(p6)
            # P7 via strided conv on P6
            p7 = self.p7(F.relu(p6))
            fpn_features.append(p7)
        
        return fpn_features


class ConvNeXtFPN(nn.Module):
    """ConvNeXt + FPN 통합 백본"""
    
    def __init__(self, model_name='tiny', pretrained=True, fpn_channels=256, use_fpn=True):
        super().__init__()
        
        self.use_fpn = use_fpn
        self.fpn_channels = fpn_channels
        
        # ConvNeXt 모델 선택 및 stage_channels 정의
        if model_name == 'tiny':
            self.convnext = convnext_tiny(pretrained=pretrained)
            self.stage_channels = [96, 192, 384, 768]
            self.extract_layers = [1, 3, 5, 7]
        elif model_name == 'small':
            self.convnext = convnext_small(pretrained=pretrained)
            self.stage_channels = [96, 192, 384, 768]
            self.extract_layers = [1, 3, 5, 7]
        elif model_name == 'base':
            self.convnext = convnext_base(pretrained=pretrained)
            self.stage_channels = [128, 256, 512, 1024]
            self.extract_layers = [1, 3, 5, 7]
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if self.use_fpn:
            # 제대로 된 FPN 사용
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=self.stage_channels,
                out_channels=fpn_channels,
                extra_blocks=None,  # 'conv' for P6, P7 if needed
                norm_layer=nn.BatchNorm2d
            )
    
    def forward_convnext(self, x):
        """ConvNeXt 특징 추출"""
        features = []
        
        # ConvNeXt stages를 통과하며 특징 추출
        for i, layer in enumerate(self.convnext.features):
            x = layer(x)
            if i in self.extract_layers:
                features.append(x)
        
        return features
    
    def forward(self, x):
        """전체 forward pass"""
        # 1. ConvNeXt로 multi-scale 특징 추출
        features = self.forward_convnext(x)
        
        if not self.use_fpn:
            # FPN 없이 마지막 특징만 반환
            return features[-1]
        
        # 2. FPN으로 특징 융합
        fpn_features = self.fpn(features)
        
        return fpn_features
    
    def get_output_channels(self):
        """출력 채널 수 반환"""
        if self.use_fpn:
            return [self.fpn_channels] * len(self.stage_channels)
        else:
            return self.stage_channels[-1]
    
    def freeze_backbone(self):
        """Backbone 파라미터 고정 (fine-tuning용)"""
        for param in self.convnext.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Backbone 파라미터 학습 가능하게"""
        for param in self.convnext.parameters():
            param.requires_grad = True