# utils/migrate_checkpoints.py
"""체크포인트 마이그레이션 유틸리티"""

import torch
from pathlib import Path
from typing import Dict, List
import shutil
from datetime import datetime


class CheckpointMigrator:
    """체크포인트 변환 및 관리"""
    
    def __init__(self, backup_dir: str = "checkpoint_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
    def migrate_checkpoint(
        self,
        old_checkpoint_path: str,
        new_checkpoint_path: str,
        key_mapping: Dict[str, str] = None,
        backup: bool = True
    ):
        """체크포인트 마이그레이션"""
        
        old_path = Path(old_checkpoint_path)
        new_path = Path(new_checkpoint_path)
        
        if not old_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {old_path}")
        
        # 백업
        if backup:
            backup_path = self.backup_dir / f"{old_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.pth"
            shutil.copy2(old_path, backup_path)
            print(f"📁 Backup saved: {backup_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(old_path, map_location='cpu')
        
        # State dict 추출
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            checkpoint = {'model_state_dict': state_dict}
        
        # 키 매핑 적용
        if key_mapping:
            new_state_dict = self._remap_keys(state_dict, key_mapping)
            checkpoint['model_state_dict'] = new_state_dict
        
        # 메타데이터 추가
        checkpoint['migration_info'] = {
            'original_path': str(old_path),
            'migrated_at': datetime.now().isoformat(),
            'key_mapping_applied': key_mapping is not None
        }
        
        # 저장
        torch.save(checkpoint, new_path)
        print(f"✅ Migrated: {old_path} -> {new_path}")
        
        return new_path
    
    def _remap_keys(self, state_dict: Dict, key_mapping: Dict[str, str]) -> Dict:
        """키 이름 변경"""
        new_state_dict = {}
        
        for old_key, value in state_dict.items():
            new_key = old_key
            
            # 정확한 매칭
            if old_key in key_mapping:
                new_key = key_mapping[old_key]
            else:
                # 접두사 매칭
                for old_prefix, new_prefix in key_mapping.items():
                    if old_key.startswith(old_prefix):
                        new_key = old_key.replace(old_prefix, new_prefix, 1)
                        break
            
            new_state_dict[new_key] = value
            
            if new_key != old_key:
                print(f"  {old_key} -> {new_key}")
        
        return new_state_dict
    
    def migrate_to_new_structure(self, checkpoint_path: str):
        """새로운 구조로 자동 변환"""
        
        # 기존 구조 -> 새 구조 매핑
        key_mapping = {
            # 백본 관련
            'backbone.convnext.': 'backbone.convnext.',
            'backbone.lateral': 'backbone.lateral_convs.',
            'backbone.smooth': 'backbone.fpn_convs.',
            
            # Head-A (블레이드)
            'blade_head.mlp': 'blade_head.mlps.',
            'segmentation_head.': 'blade_head.',  # 옛날 이름
            
            # Head-B (손상)
            'damage_head.fusion': 'damage_head.seg_fusion',
            'damage_detection_head.': 'damage_head.',  # 옛날 이름
        }
        
        new_path = checkpoint_path.replace('.pth', '_v2.pth')
        return self.migrate_checkpoint(checkpoint_path, new_path, key_mapping)
    
    def batch_migrate(self, checkpoint_list: List[str]):
        """여러 체크포인트 일괄 변환"""
        
        results = []
        for checkpoint_path in checkpoint_list:
            try:
                new_path = self.migrate_to_new_structure(checkpoint_path)
                results.append((checkpoint_path, new_path, "Success"))
            except Exception as e:
                results.append((checkpoint_path, None, str(e)))
                print(f"❌ Failed: {checkpoint_path} - {e}")
        
        # 결과 요약
        print("\n" + "="*50)
        print("Migration Summary:")
        print("="*50)
        
        for old, new, status in results:
            if new:
                print(f"✅ {old} -> {new}")
            else:
                print(f"❌ {old}: {status}")
        
        return results


# 간단한 함수 버전
def quick_migrate(old_path: str, new_path: str = None):
    """빠른 마이그레이션"""
    
    if new_path is None:
        new_path = old_path.replace('.pth', '_migrated.pth')
    
    migrator = CheckpointMigrator()
    return migrator.migrate_to_new_structure(old_path)