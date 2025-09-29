# models/backbones/__init__.py
from .convnext_fpn import ConvNeXtFPN

def build_backbone(
    backbone_type='convnext_fpn',
    model_size='tiny',
    pretrained=True,
    **kwargs
):
    """백본 생성 헬퍼 함수"""
    
    if backbone_type == 'convnext_fpn':
        return ConvNeXtFPN(
            model_name=model_size,
            pretrained=pretrained,
            use_fpn=True,
            **kwargs
        )
    
    elif backbone_type == 'convnext_only':
        return ConvNeXtFPN(
            model_name=model_size,
            pretrained=pretrained,
            use_fpn=False,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")

__all__ = ['ConvNeXtFPN', 'SimplifiedFPN', 'build_backbone']