# setup_env.py
import sys
import os
import warnings
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.settings import settings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_opencv():
    """OpenCV 초기화"""
    import cv2
    
    # 필요한 상수 패치
    cv2_constants = {
        'COLOR_BGR2RGB': 4,
        'COLOR_RGB2BGR': 5,
        'COLOR_BGR2GRAY': 6,
        'COLOR_BGR2HSV': 40,
        'COLOR_HSV2BGR': 54,
    }
    
    for name, value in cv2_constants.items():
        if not hasattr(cv2, name):
            setattr(cv2, name, value)
    
    cv2.setNumThreads(settings.opencv_num_threads)
    logger.info(f"OpenCV {cv2.__version__} initialized")
    return cv2

def setup_cuda():
    """CUDA 환경 설정"""
    import torch
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(settings.cuda_device)
    torch.backends.cudnn.benchmark = settings.cudnn_benchmark
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    return torch

def initialize():
    """환경 초기화"""
    warnings.filterwarnings("ignore")
    
    # OpenCV 설정
    cv2 = setup_opencv()
    
    # CUDA 설정
    torch = setup_cuda()
    
    # MM 라이브러리
    import mmcv
    import mmdet
    import mmseg
    
    logger.info(f"Environment initialized")
    logger.info(f"Mode: {settings.inspection_mode}")
    logger.info(f"Keyframe interval: {settings.keyframe_interval}")
    
    return {
        'cv2': cv2,
        'torch': torch,
        'mmcv': mmcv,
        'mmdet': mmdet,
        'mmseg': mmseg
    }

# import 시 자동 실행
modules = initialize()