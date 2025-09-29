# train_unified.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from models.unified.unified_model import UnifiedModel
from utils.dataset import UnifiedDamageDataset, create_dataloaders
from utils.evaluate import ModelEvaluator


class UnifiedTrainer:
    """Trainer for UnifiedModel with mixed precision and gradient accumulation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        self._setup_model()
        
        # Create datasets and dataloaders
        self._setup_data()
        
        # Setup training components
        self._setup_training()
        
        # Setup logging
        self._setup_logging()
        
    def _setup_model(self):
        """Initialize unified model"""
        model_config = {
            'backbone_type': self.config.get('backbone_type', 'tiny'),
            'use_fpn': self.config.get('use_fpn', True),
            'num_blade_classes': 1,  # Binary segmentation
            'num_damage_classes': 3,  # crack, nick, tear
            'use_hungarian': self.config.get('use_hungarian', False),
            'mask2former_config': self.config.get('mask2former_config', {
                'queries_per_class': 100,  # 클래스당 100개
                'hidden_dim': 256,
                'num_heads': 8,
                'dec_layers': 3,
                'dropout': 0.1,
                'use_blade_mask': True
            })
        }
        
        self.model = UnifiedModel(model_config).to(self.device)
        
        # Load checkpoint if provided
        if 'checkpoint' in self.config:
            self.load_checkpoint(self.config['checkpoint'])
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")
        
    def _setup_data(self):
        """Setup datasets and dataloaders"""
        self.train_loader, self.valid_loader, self.test_loader = create_dataloaders(
            blade_root=self.config['blade_data_root'],
            damage_root=self.config['damage_data_root'],
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 4),
            use_instance_masks=self.config.get('use_hungarian', False)
        )
        
        print(f"Dataset sizes - Train: {len(self.train_loader.dataset)}, "
              f"Valid: {len(self.valid_loader.dataset)}, "
              f"Test: {len(self.test_loader.dataset)}")
        
    def _setup_training(self):
        """Setup optimizer, scheduler, and scaler"""
        
        # Get parameter groups with different learning rates
        param_groups = self.model.get_param_groups(
            base_lr=self.config['learning_rate']
        )
        
        # Optimizer
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config.get('weight_decay', 0.05)
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config['epochs']
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.get('use_amp', True) else None
        
        # Gradient accumulation
        self.accumulate_steps = self.config.get('accumulate_grad_batches', 1)
        
    def _setup_logging(self):
        """Setup logging (WandB or tensorboard)"""
        
        self.use_wandb = self.config.get('use_wandb', False)
        
        if self.use_wandb:
            wandb.init(
                project="blade-damage-detection",
                name=self.config.get('experiment_name', 'unified_model'),
                config=self.config
            )
            wandb.watch(self.model, log_freq=100)
        
        # Create log file
        self.log_file = self.output_dir / 'training_log.json'
        self.training_history = []
        
    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0, 'blade_loss': 0, 'damage_loss': 0,
            'multilabel_loss': 0, 'mask_loss': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            blade_masks = batch['blade_mask'].to(self.device)
            damage_masks = batch['damage_mask'].to(self.device)
            multilabels = batch['multilabel'].to(self.device)
            
            # Prepare targets
            targets = {
                'blade_mask': blade_masks,
                'damage_mask': damage_masks,
                'multilabel': multilabels
            }
            
            if batch.get('instance_masks') is not None:
                targets['instance_masks'] = batch['instance_masks']
                targets['instance_labels'] = batch['instance_labels']
            
            # Forward pass with mixed precision
            with autocast(enabled=self.scaler is not None):
                outputs, losses = self.model(images, targets)
                total_loss = losses['total']
                
                # Scale loss for gradient accumulation
                total_loss = total_loss / self.accumulate_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.accumulate_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip', 0.01)
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip', 0.01)
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            # Update metrics
            for key in losses:
                if key in epoch_metrics:
                    epoch_metrics[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'blade': f"{losses.get('blade_loss', 0):.4f}",
                'damage': f"{losses.get('damage_loss', 0):.4f}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    f"train/{k}": v.item() if torch.is_tensor(v) else v
                    for k, v in losses.items()
                })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.train_loader)
        
        return epoch_metrics
    
    def validate(self):
        """Validation step"""
        self.model.eval()
        
        val_metrics = {
            'loss': 0, 'blade_iou': 0, 'damage_f1': 0,
            'multilabel_accuracy': 0
        }
        
        evaluator = ModelEvaluator(self.model, self.device)
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="Validation"):
                # Move to device
                images = batch['image'].to(self.device)
                blade_masks = batch['blade_mask'].to(self.device)
                damage_masks = batch['damage_mask'].to(self.device)
                multilabels = batch['multilabel'].to(self.device)
                
                # Prepare targets
                targets = {
                    'blade_mask': blade_masks,
                    'damage_mask': damage_masks,
                    'multilabel': multilabels
                }
                
                # Forward pass
                with autocast(enabled=self.scaler is not None):
                    outputs, losses = self.model(images, targets)
                
                # Compute metrics
                batch_metrics = evaluator.compute_batch_metrics(
                    outputs, targets
                )
                
                # Update metrics
                val_metrics['loss'] += losses['total'].item()
                for key in batch_metrics:
                    if key in val_metrics:
                        val_metrics[key] += batch_metrics[key]
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= len(self.valid_loader)
        
        return val_metrics
    
    def train(self):
        """Main training loop"""
        best_f1 = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'='*50}")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            print(f"Train - Loss: {train_metrics['loss']:.4f}")
            
            # Validation
            val_metrics = self.validate()
            print(f"Valid - Loss: {val_metrics['loss']:.4f}, "
                  f"Blade IoU: {val_metrics['blade_iou']:.4f}, "
                  f"Damage F1: {val_metrics['damage_f1']:.4f}")
            
            # Save checkpoint
            if val_metrics['damage_f1'] > best_f1:
                best_f1 = val_metrics['damage_f1']
                self.save_checkpoint(
                    epoch,
                    val_metrics,
                    is_best=True
                )
                print(f"✅ New best model saved (F1: {best_f1:.4f})")
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(epoch, val_metrics)
            
            # Log history
            self.training_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'valid': val_metrics
            })
            
            # Save training history
            with open(self.log_file, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"valid/{k}": v for k, v in val_metrics.items()}
                })
        
        print(f"\n🎉 Training completed! Best F1: {best_f1:.4f}")
        
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"   Metrics: {checkpoint['metrics']}")


def main():
    """Main function"""
    
    # Configuration
    config = {
        # Data paths
        'blade_data_root': 'C:/EngineBladeAI/EngineInspectionAI_MS/data/blade_data',
        'damage_data_root': 'C:/EngineBladeAI/EngineInspectionAI_MS/data/multilabeled_data_augmented',
        
        # Model config
        'backbone_type': 'tiny',
        'use_fpn': True,
        'use_hungarian': False,  # Start with SimpleLoss
        
        'mask2former_config': {
            'queries_per_class': 100,  # 클래스당 100개
            'hidden_dim': 256,
            'num_heads': 8,
            'dec_layers': 3,
            'dropout': 0.1,
            'use_blade_mask': True
        },
        
        # Training config
        'batch_size': 2,
        'accumulate_grad_batches': 2,
        'num_workers': 4,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'gradient_clip': 0.01,
        'use_amp': True,
        
        # Logging
        'output_dir': 'outputs/unified_training',
        'experiment_name': f'unified_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'use_wandb': False,
        'save_every': 5
    }
    
    # Create trainer and train
    trainer = UnifiedTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()