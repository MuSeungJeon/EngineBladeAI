# utils/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """YAML 설정 파일 로더"""
    
    @staticmethod
    def load_yaml(yaml_path: str) -> Dict[str, Any]:
        """YAML 파일 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # inherit_from 처리
        if 'inherit_from' in config:
            parent_config = ConfigLoader.load_yaml(config['inherit_from'])
            config = ConfigLoader.merge_configs(parent_config, config)
        
        return config
    
    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """설정 병합 (override가 우선)"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def load_experiment_config(exp_name: str) -> Dict:
        """실험 설정 전체 로드"""
        exp_config = ConfigLoader.load_yaml(f"configs/experiments/{exp_name}.yaml")
        
        # 각 설정 파일 로드
        model_config = ConfigLoader.load_yaml(exp_config['model'])
        train_config = ConfigLoader.load_yaml(exp_config['training'])
        data_config = ConfigLoader.load_yaml(exp_config['data'])
        
        return {
            'experiment': exp_config['experiment'],
            'model': model_config,
            'training': train_config,
            'data': data_config,
            'hyperparameters': exp_config.get('hyperparameters', {})
        }

# 사용 예시
if __name__ == "__main__":
    config = ConfigLoader.load_yaml("configs/train_configs/unified.yaml")
    print(yaml.dump(config, default_flow_style=False))