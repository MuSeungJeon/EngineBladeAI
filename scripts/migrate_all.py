# scripts/migrate_all.py
"""모든 체크포인트 마이그레이션 실행"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from utils.migrate_checkpoints import CheckpointMigrator


def main():
    """메인 실행 함수"""
    
    # 마이그레이션할 체크포인트 목록
    checkpoints = [
        'best_unified_blade_model.pth',
        'best_damage_model.pth',
        'best_deepLab_combined.pth',
        'best_segformer_model.pth'
    ]
    
    # 마이그레이터 생성
    migrator = CheckpointMigrator(backup_dir='checkpoint_backups')
    
    # 존재하는 체크포인트만 필터링
    existing_checkpoints = []
    for cp in checkpoints:
        if Path(cp).exists():
            existing_checkpoints.append(cp)
            print(f"Found: {cp}")
        else:
            print(f"Not found: {cp}")
    
    if not existing_checkpoints:
        print("No checkpoints to migrate!")
        return
    
    # 일괄 마이그레이션
    print("\nStarting migration...")
    print("="*50)
    
    results = migrator.batch_migrate(existing_checkpoints)
    
    print("\n✅ Migration complete!")


if __name__ == "__main__":
    main()