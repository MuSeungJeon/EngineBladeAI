# unified_trainer_jupyter.py
"""
주피터 노트북용 통합 모델 학습기 - 간단하고 모듈화된 버전
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional

# 간단한 Hungarian Loss (이전 버전 사용)
class SimpleHungarianLoss(nn.Module):
    def __init__(self, num_classes=3, weight_dict=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcherFixed()
        
        # Loss weights
        self.weight_dict = weight_dict or {
            'loss_ce': 2.0,
            'loss_mask': 5.0,
            'loss_dice': 2.0
        }
        
    def forward(self, outputs, batch):
        # Target 형식 맞추기
        targets = {
            'instance_masks': batch['instance_masks'],
            'instance_labels': batch['instance_labels']
        }
        
        # Hungarian matching
        indices = self.matcher(outputs, targets)
        
        # 각 loss 계산
        losses = {}
        losses['ce'] = self.loss_labels(outputs, targets, indices)
        losses['mask'] = self.loss_masks(outputs, targets, indices)
        losses['dice'] = self.loss_dice(outputs, targets, indices)
        
        # Total loss
        total_loss = sum(losses[k] * self.weight_dict[f'loss_{k}'] 
                        for k in losses.keys())
        
        return total_loss, losses
    
    def loss_labels(self, outputs, targets, indices):
        """Classification loss"""
        src_logits = outputs['pred_logits']  # [B, Q, C]
        
        # Matched pairs만 loss 계산
        loss = 0
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
                
            pred = src_logits[i, src_idx]  # [matched_q, C]
            tgt = targets['instance_labels'][i][tgt_idx]  # [matched_q]
            
            loss += nn.functional.cross_entropy(pred, tgt)
            
        return loss / len(indices) if len(indices) > 0 else torch.tensor(0.0)
    
    def loss_masks(self, outputs, targets, indices):
        """Mask loss (BCE)"""
        src_masks = outputs['pred_masks']
        
        # 크기 맞추기
        if src_masks.shape[-2:] != targets['instance_masks'][0].shape[-2:]:
            src_masks = nn.functional.interpolate(
                src_masks,
                size=targets['instance_masks'][0].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        loss = 0
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
                
            pred = src_masks[i, src_idx]
            tgt = targets['instance_masks'][i][tgt_idx].float()
            
            loss += nn.functional.binary_cross_entropy_with_logits(
                pred.flatten(1), 
                tgt.flatten(1)
            )
            
        return loss / len(indices) if len(indices) > 0 else torch.tensor(0.0)
    
    def loss_dice(self, outputs, targets, indices):
        """Dice loss"""
        src_masks = outputs['pred_masks']
        
        # 크기 맞추기
        if src_masks.shape[-2:] != targets['instance_masks'][0].shape[-2:]:
            src_masks = nn.functional.interpolate(
                src_masks,
                size=targets['instance_masks'][0].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        loss = 0
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
                
            pred = src_masks[i, src_idx].sigmoid()
            tgt = targets['instance_masks'][i][tgt_idx].float()
            
            numerator = 2 * (pred * tgt).sum(dim=[1, 2])
            denominator = pred.sum(dim=[1, 2]) + tgt.sum(dim=[1, 2])
            
            loss += (1 - (numerator + 1) / (denominator + 1)).mean()
            
        return loss / len(indices) if len(indices) > 0 else torch.tensor(0.0)


class JupyterUnifiedTrainer:
    """주피터용 간단한 통합 모델 학습기"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Device: {self.device}")
        
        # 모델 생성
        self.model = self._create_model()
        
        # Loss functions
        self.blade_criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.damage_criterion = SimpleHungarianLoss(num_classes=3)
        self.multilabel_criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'blade_iou': [],
            'damage_f1': [],
            'lr': []
        }
        
        print("✅ Trainer initialized!")
    
    def _create_model(self):
        """모델 생성"""
        from models.unified.unified_model_v2 import UnifiedModelV2
        
        model = UnifiedModelV2(
            backbone_type=self.config.get('backbone_type', 'tiny'),
            pretrained_backbone=self.config.get('pretrained_backbone', True),
            fpn_channels=self.config.get('fpn_channels', 256),
            num_blade_classes=self.config.get('num_blade_classes', 2),
            num_damage_classes=self.config.get('num_damage_classes', 3),
            num_queries=self.config.get('num_queries', 100),
            dec_layers=self.config.get('dec_layers', 3),
            freeze_blade=self.config.get('freeze_blade', True),
            use_hungarian=False,  # 우리가 직접 처리
            use_dynamic_balancing=False
        )
        
        model = model.to(self.device)
        
        # 모델 정보 출력
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"📊 Model: {total_params:.1f}M total, {trainable_params:.1f}M trainable")
        
        return model
    
    def _create_optimizer(self):
        """옵티마이저 생성"""
        lr = self.config.get('learning_rate', 2e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        return optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    
    def _create_scheduler(self):
        """스케줄러 생성"""
        if self.config.get('use_scheduler', True):
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 10),
                eta_min=self.config.get('learning_rate', 2e-4) * 0.01
            )
        return None
    
    def train_step(self, batch) -> Dict[str, float]:
        """한 스텝 학습"""
        self.model.train()
        
        # Move to device
        images = batch['image'].to(self.device)
        blade_masks = batch['blade_mask'].to(self.device)
        multilabels = batch['multilabel'].to(self.device)
        
        # Forward pass
        if self.scaler:
            with autocast():
                outputs = self.model(images, return_loss=False)
                
                # Calculate losses
                blade_loss = self.blade_criterion(outputs['blade_logits'], blade_masks)
                multilabel_loss = self.multilabel_criterion(outputs['multilabel'], multilabels)
                
                # Hungarian loss
                damage_loss, damage_losses = self.damage_criterion(outputs, batch)
                
                # Total loss
                total_loss = blade_loss + damage_loss + multilabel_loss
        else:
            outputs = self.model(images, return_loss=False)
            
            # Calculate losses
            blade_loss = self.blade_criterion(outputs['blade_logits'], blade_masks)
            multilabel_loss = self.multilabel_criterion(outputs['multilabel'], multilabels)
            
            # Hungarian loss
            damage_loss, damage_losses = self.damage_criterion(outputs, batch)
            
            # Total loss
            total_loss = blade_loss + damage_loss + multilabel_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()
        
        # Return losses
        return {
            'total_loss': total_loss.item(),
            'blade_loss': blade_loss.item(),
            'damage_loss': damage_loss.item(),
            'multilabel_loss': multilabel_loss.item(),
            **{k: v.item() if torch.is_tensor(v) else v for k, v in damage_losses.items()}
        }
    
    def validate_step(self, batch) -> Dict[str, float]:
        """검증 스텝"""
        self.model.eval()
        
        with torch.no_grad():
            images = batch['image'].to(self.device)
            blade_masks = batch['blade_mask'].to(self.device)
            multilabels = batch['multilabel'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, return_loss=False)
            
            # Calculate losses
            blade_loss = self.blade_criterion(outputs['blade_logits'], blade_masks)
            multilabel_loss = self.multilabel_criterion(outputs['multilabel'], multilabels)
            damage_loss, damage_losses = self.damage_criterion(outputs, batch)
            total_loss = blade_loss + damage_loss + multilabel_loss
            
            # Calculate metrics
            blade_pred = outputs['blade_logits'].argmax(1)
            blade_iou = self._calculate_iou(blade_pred, blade_masks)
            
            damage_pred = (outputs['multilabel'] > 0.5).float()
            damage_f1 = self._calculate_f1(damage_pred, multilabels)
            
            return {
                'total_loss': total_loss.item(),
                'blade_loss': blade_loss.item(),
                'damage_loss': damage_loss.item(),
                'multilabel_loss': multilabel_loss.item(),
                'blade_iou': blade_iou,
                'damage_f1': damage_f1,
                **{k: v.item() if torch.is_tensor(v) else v for k, v in damage_losses.items()}
            }
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """한 에폭 학습"""
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            step_metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{step_metrics['total_loss']:.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] = np.mean(epoch_metrics[key])
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """한 에폭 검증"""
        epoch_metrics = {}
        
        for batch in tqdm(val_loader, desc='Validation'):
            step_metrics = self.validate_step(batch)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] = np.mean(epoch_metrics[key])
        
        return epoch_metrics
    
    def train(self, train_loader, val_loader, epochs: int = 10):
        """전체 학습 루프"""
        print(f"🔥 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Update history
            self.metrics_history['train_loss'].append(train_metrics['total_loss'])
            self.metrics_history['val_loss'].append(val_metrics['total_loss'])
            self.metrics_history['blade_iou'].append(val_metrics['blade_iou'])
            self.metrics_history['damage_f1'].append(val_metrics['damage_f1'])
            self.metrics_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print results
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"Blade IoU: {val_metrics['blade_iou']:.4f}")
            print(f"Damage F1: {val_metrics['damage_f1']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        print("✅ Training completed!")
    
    def _calculate_iou(self, pred, target):
        """IoU 계산"""
        intersection = (pred & target).float().sum()
        union = (pred | target).float().sum()
        iou = intersection / (union + 1e-8)
        return iou.item()
    
    def _calculate_f1(self, pred, target):
        """F1 점수 계산"""
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return f1.item()
    
    def plot_metrics(self, figsize=(15, 5)):
        """메트릭 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss curves
        axes[0].plot(self.metrics_history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.metrics_history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Performance metrics
        axes[1].plot(self.metrics_history['blade_iou'], label='Blade IoU', marker='o')
        axes[1].plot(self.metrics_history['damage_f1'], label='Damage F1', marker='s')
        axes[1].set_title('Performance Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning rate
        axes[2].plot(self.metrics_history['lr'], label='Learning Rate', marker='o')
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_checkpoint(self, path: str):
        """체크포인트 저장"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.metrics_history = checkpoint['metrics_history']
        
        print(f"📂 Checkpoint loaded from {path}")


# 간단한 설정 생성 함수
def create_simple_config(
    backbone_type='tiny',
    learning_rate=2e-4,
    batch_size=2,
    use_amp=True,
    freeze_blade=True
):
    """간단한 설정 생성"""
    return {
        'backbone_type': backbone_type,
        'pretrained_backbone': True,
        'fpn_channels': 256,
        'num_blade_classes': 2,
        'num_damage_classes': 3,
        'num_queries': 100,
        'dec_layers': 3,
        'learning_rate': learning_rate,
        'weight_decay': 1e-4,
        'batch_size': batch_size,
        'use_amp': use_amp,
        'use_scheduler': True,
        'freeze_blade': freeze_blade,
        'epochs': 10
    }


# 빠른 테스트 함수
def quick_test():
    """빠른 모델 테스트"""
    print("🧪 Quick model test...")
    
    config = create_simple_config()
    trainer = JupyterUnifiedTrainer(config)
    
    # 더미 데이터
    dummy_batch = {
        'image': torch.randn(2, 3, 640, 640),
        'blade_mask': torch.randint(0, 2, (2, 640, 640)),
        'multilabel': torch.randint(0, 2, (2, 3)).float(),
        'instance_masks': [torch.randint(0, 2, (2, 640, 640)).float(), 
                          torch.randint(0, 2, (1, 640, 640)).float()],
        'instance_labels': [torch.tensor([0, 1]), torch.tensor([2])]
    }
    
    # Forward test
    metrics = trainer.validate_step(dummy_batch)
    print("✅ Forward pass successful!")
    print(f"Metrics: {metrics}")
    
    return trainer


if __name__ == "__main__":
    quick_test()
