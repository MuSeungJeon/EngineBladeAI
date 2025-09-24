# models/backbones/convnext_fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, convnext_small, convnext_base


class ConvNeXtFPN(nn.Module):
    """ConvNeXt + FPN 통합 백본"""
    
    def __init__(self, model_name='tiny', pretrained=True, fpn_channels=256, use_fpn=True):
        super().__init__()
        
        self.use_fpn = use_fpn
        self.fpn_channels = fpn_channels
        
        if model_name == 'tiny':
            self.convnext = convnext_tiny(pretrained=pretrained)
            # ConvNeXt-Tiny의 실제 출력 채널
            self.stage_channels = [192, 384, 768, 768]
            self.extract_layers = [2, 4, 6, 7]
        
        if self.use_fpn:
            # FPN Lateral connections - 채널 수 수정
            self.lateral_convs = nn.ModuleList()
            for in_channels in self.stage_channels:  # [192, 384, 768, 768]
                self.lateral_convs.append(
                    nn.Conv2d(in_channels, fpn_channels, kernel_size=1)
                )
            
            # FPN Output convolutions
            self.fpn_convs = nn.ModuleList()
            for _ in range(len(self.stage_channels)):
                self.fpn_convs.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(fpn_channels),
                        nn.ReLU(inplace=True)
                    )
                )
            
            # 4. Extra FPN levels (P6, P7) - 선택사항
            self.extra_levels = False
            if self.extra_levels:
                self.p6 = nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1)
                self.p7 = nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1)
    
    def forward_convnext(self, x):
        """ConvNeXt 특징 추출"""
        features = []
        
        # ConvNeXt stages를 통과하며 특징 추출
        for i, layer in enumerate(self.convnext.features):
            x = layer(x)
            if i in self.extract_layers:
                features.append(x)
        
        return features
    
    def forward_fpn(self, features):
        """FPN 처리"""
        # Bottom-up pathway는 이미 완료 (features)
        
        # Lateral connections
        laterals = []
        for feat, lateral_conv in zip(features, self.lateral_convs):
            laterals.append(lateral_conv(feat))
        
        # Top-down pathway + lateral connections
        # 가장 작은 해상도(P4)부터 시작
        fpn_features = []
        
        # P4 (가장 작은 해상도, 가장 깊은 특징)
        fpn_features.append(laterals[-1])
        
        # P3, P2, P1 (큰 해상도로 가면서)
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample from previous level
            upsampled = F.interpolate(
                fpn_features[0],  # 이전 레벨 (더 작은 해상도)
                size=laterals[i].shape[-2:],
                mode='nearest'
            )
            
            # Add lateral connection
            fpn_feat = laterals[i] + upsampled
            fpn_features.insert(0, fpn_feat)  # 앞에 추가 (순서 유지)
        
        # Apply 3x3 conv to each FPN level
        fpn_outputs = []
        for feat, fpn_conv in zip(fpn_features, self.fpn_convs):
            fpn_outputs.append(fpn_conv(feat))
        
        # Extra levels if needed
        if self.extra_levels:
            p6 = self.p6(fpn_outputs[-1])
            p7 = self.p7(F.relu(p6))
            fpn_outputs.extend([p6, p7])
        
        return fpn_outputs
    
    def forward(self, x):
        """전체 forward pass"""
        # 1. ConvNeXt로 multi-scale 특징 추출
        features = self.forward_convnext(x)
        
        if not self.use_fpn:
            # FPN 없이 마지막 특징만 반환
            return features[-1]
        
        # 2. FPN으로 특징 융합
        fpn_features = self.forward_fpn(features)
        
        return fpn_features
    
    def get_output_channels(self):
        """출력 채널 수 반환"""
        if self.use_fpn:
            return [self.fpn_channels] * len(self.stage_channels)
        else:
            return self.stage_channels[-1]


class SimplifiedFPN(nn.Module):
    """간단한 FPN 구현 (빠른 테스트용)"""
    
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        self.laterals = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])
        
        self.smooths = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features):
        # Build laterals
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]
        
        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            size = laterals[i-1].shape[-2:]
            laterals[i-1] += F.interpolate(laterals[i], size=size, mode='nearest')
        
        # Smooth
        outputs = [smooth(lat) for smooth, lat in zip(self.smooths, laterals)]
        
        return outputs