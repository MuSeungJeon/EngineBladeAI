# config/settings.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class Settings:
    """블레이드 검사 시스템 전체 설정"""
    
    # ========== 경로 설정 ==========
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_root: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    model_root: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    output_root: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs")
    log_root: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # ========== 시스템 설정 ==========
    cuda_device: int = 0
    mixed_precision: bool = True
    cudnn_benchmark: bool = True
    opencv_num_threads: int = 4
    num_workers: int = 4
    
    # ========== 모델 설정 ==========
    backbone: str = "ConvNeXt-Tiny"
    num_damage_classes: int = 5  # Tear, Nick, Crack, Dent, Coating
    num_queries: int = 100
    
    # ========== 학습 설정 ==========
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 100
    
    # 손실 가중치
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'blade': 1.0,
        'damage': 1.0,
        'outside': 0.5,
        'size': 0.2,
        'temporal': 0.3
    })
    
    # ========== 추론 설정 ==========
    inspection_mode: str = "auto"  # "auto" or "manual"
    keyframe_interval: int = 5
    force_refresh_interval: int = 50
    
    # 블러 임계값
    blur_threshold_high: int = 150
    blur_threshold_low: int = 80
    max_consecutive_skip: int = 5
    
    # ROI 설정
    max_rois_per_frame: int = 8
    max_pixels_per_frame: int = 2_000_000
    
    # 품질 체크
    temporal_iou_threshold: float = 0.6
    confidence_threshold: float = 0.5
    
    # 히스테리시스 (회전 속도)
    rotation_fast_threshold: float = 180.0  # deg/s
    rotation_slow_threshold: float = 120.0  # deg/s
    
    # ========== 데이터 설정 ==========
    input_size: tuple = (1024, 1024)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    def __post_init__(self):
        """디렉토리 자동 생성"""
        for path in [self.data_root, self.model_root, self.output_root, self.log_root]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save(self, path: str):
        """설정 저장"""
        import json
        from dataclasses import asdict
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str):
        """설정 로드"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# 전역 설정 인스턴스
settings = Settings()