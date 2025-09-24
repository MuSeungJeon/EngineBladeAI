# configs/training_config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    """학습 설정 (모든 모델 공통)"""
    
    # Model selection
    model_type: str = 'cnn'  # 'cnn' or 'mask2former'
    
    # Mask2Former specific
    mask2former_config = {
        'batch_size': 2,
        'accumulate_grad_batches': 2,
        'num_queries': 100,
        'dec_layers': 3,
        'use_amp': True,
        'learning_rate': 1e-5,
        'gradient_clip': 0.01
    }
    
    # CNN specific
    cnn_config = {
        'batch_size': 4,
        'accumulate_grad_batches': 1,
        'use_soft_gating': True,
        'boundary_margin': 10,
        'learning_rate': 1e-4,
        'gradient_clip': 1.0,
        'use_amp': False
    }
    
    def get_config(self):
        if self.model_type == 'mask2former':
            return self.mask2former_config
        return self.cnn_config