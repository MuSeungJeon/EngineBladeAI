# utils/train.py - Mask2Former 지원 버전
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional
from torch.cuda.amp import GradScaler, autocast


def train_damage_model(
    model, 
    train_loader, 
    val_loader, 
    epochs=20, 
    lr=1e-4, 
    device='cuda',
    model_type='cnn',  # 'cnn' or 'mask2former'
    use_amp=False,
    accumulate_grad_batches=1
):
    """손상 검출 모델 학습 (CNN/Mask2Former 모두 지원)"""
    
    model.to(device)
    
    # Model type에 따른 설정
    if model_type == 'mask2former':
        from models.heads.mask2former_damage_head import Mask2FormerLoss
        criterion = Mask2FormerLoss(
            num_classes=3,
            matcher_cost_class=2.0,
            matcher_cost_mask=5.0,
            matcher_cost_dice=5.0,
            aux_loss=True,
            aux_weight=0.4
        )
        lr = lr * 0.1  # Mask2Former는 더 작은 lr
        gradient_clip = 0.01
    else:
        # CNN 기반
        criterion = CombinedDamageLoss()
        gradient_clip = 1.0
    
    # Optimizer 설정
    param_groups = get_param_groups(model, lr, model_type)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    
    # Scheduler
    scheduler = get_scheduler(optimizer, epochs, len(train_loader))
    
    # Mixed precision
    scaler = GradScaler() if use_amp else None
    
    # Training history
    best_val_score = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': []
    }
    
    for epoch in range(epochs):
        # Training phase
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, epochs, gradient_clip, scaler, 
            accumulate_grad_batches, model_type
        )
        
        # Validation phase
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, model_type
        )
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Print epoch summary
        print_epoch_summary(epoch, epochs, train_metrics, val_metrics)
        
        # Save best model
        score = get_score(val_metrics, model_type)
        if score > best_val_score:
            best_val_score = score
            save_checkpoint(model, optimizer, scheduler, epoch, 
                          val_metrics, best_val_score, model_type)
            print(f"  ✅ Best model saved (Score: {best_val_score:.4f})")
    
    return history, best_val_score


def train_epoch(
    model, train_loader, criterion, optimizer, scheduler,
    device, epoch, total_epochs, gradient_clip, scaler,
    accumulate_grad_batches, model_type
):
    """한 에폭 학습"""
    model.train()
    
    total_loss = 0
    loss_components = {}
    metrics_accumulator = MetricsAccumulator(model_type)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = move_batch_to_device(batch, device)
        
        # Mixed precision training
        with autocast(enabled=(scaler is not None)):
            # Forward pass
            if model_type == 'mask2former':
                outputs = model(batch['image'], batch.get('blade_mask'))
                loss, loss_dict = criterion(outputs, batch)
            else:
                outputs = model(batch['image'], batch.get('blade_mask'))
                loss, loss_dict = criterion(outputs, batch)
        
        # Gradient accumulation
        loss = loss / accumulate_grad_batches
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if (batch_idx + 1) % accumulate_grad_batches == 0:
            if scaler:
                scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        # Update metrics
        total_loss += loss.item() * accumulate_grad_batches
        metrics_accumulator.update(outputs, batch)
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulate_grad_batches:.4f}',
            'lr': f'{current_lr:.6f}'
        })
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    metrics = metrics_accumulator.compute()
    metrics['loss'] = avg_loss
    
    return metrics


def validate_epoch(model, val_loader, criterion, device, model_type):
    """검증 에폭"""
    model.eval()
    
    total_loss = 0
    metrics_accumulator = MetricsAccumulator(model_type)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            batch = move_batch_to_device(batch, device)
            
            # Forward pass
            outputs = model(batch['image'], batch.get('blade_mask'))
            
            # Calculate loss
            if model_type == 'mask2former':
                loss, _ = criterion(outputs, batch)
            else:
                loss, _ = criterion(outputs, batch)
            
            total_loss += loss.item()
            metrics_accumulator.update(outputs, batch)
    
    # Calculate metrics
    metrics = metrics_accumulator.compute()
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics


class MetricsAccumulator:
    """메트릭 누적 계산"""
    
    def __init__(self, model_type='cnn'):
        self.model_type = model_type
        self.reset()
    
    def reset(self):
        self.tp = torch.zeros(3)
        self.fp = torch.zeros(3)
        self.fn = torch.zeros(3)
        self.ious = []
        self.dices = []
        
        if self.model_type == 'mask2former':
            self.ap_scores = []
            self.query_accuracies = []
    
    def update(self, outputs, targets):
        """메트릭 업데이트"""
        if self.model_type == 'mask2former':
            self._update_mask2former(outputs, targets)
        else:
            self._update_cnn(outputs, targets)
    
    def _update_cnn(self, outputs, targets):
        """CNN 모델 메트릭"""
        if 'multilabel' in outputs:
            pred_ml = (outputs['multilabel'] > 0.5).float().cpu()
            target_ml = targets['multilabel'].cpu()
            
            self.tp += (pred_ml * target_ml).sum(dim=0)
            self.fp += (pred_ml * (1 - target_ml)).sum(dim=0)
            self.fn += ((1 - pred_ml) * target_ml).sum(dim=0)
        
        if 'segmentation' in outputs:
            pred_seg = outputs['segmentation'].argmax(1).cpu()
            target_seg = targets['damage_mask'].cpu()
            
            intersection = (pred_seg & target_seg).float().sum()
            union = (pred_seg | target_seg).float().sum()
            self.ious.append((intersection / (union + 1e-6)).item())
    
    def _update_mask2former(self, outputs, targets):
        """Mask2Former 메트릭"""
        # Instance-level metrics
        pred_masks = outputs['pred_masks']  # [B, num_queries, H, W]
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
        
        # Top-k predictions
        scores = pred_logits.softmax(-1)[..., :-1].max(-1)[0]
        
        # Multi-label from aggregated queries
        if 'multilabel' in outputs:
            pred_ml = (outputs['multilabel'] > 0.5).float().cpu()
            target_ml = targets['multilabel'].cpu()
            
            self.tp += (pred_ml * target_ml).sum(dim=0)
            self.fp += (pred_ml * (1 - target_ml)).sum(dim=0)
            self.fn += ((1 - pred_ml) * target_ml).sum(dim=0)
        
        # Query accuracy
        if 'instance_labels' in targets:
            # Match queries to targets and calculate accuracy
            matched_accuracy = self._calculate_query_matching(
                pred_logits, pred_masks, targets
            )
            self.query_accuracies.append(matched_accuracy)
    
    def compute(self):
        """최종 메트릭 계산"""
        metrics = {}
        
        # F1 scores
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics['precision'] = precision.numpy()
        metrics['recall'] = recall.numpy()
        metrics['f1'] = f1.numpy()
        metrics['avg_f1'] = f1.mean().item()
        
        # Segmentation metrics
        if self.ious:
            metrics['iou'] = np.mean(self.ious)
        
        # Mask2Former specific
        if self.model_type == 'mask2former' and self.query_accuracies:
            metrics['query_acc'] = np.mean(self.query_accuracies)
        
        return metrics


class CombinedDamageLoss(nn.Module):
    """CNN 모델용 복합 손실"""
    
    def __init__(self):
        super().__init__()
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.ml_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, outputs, targets):
        losses = {}
        total_loss = 0
        
        # Segmentation loss
        if 'segmentation' in outputs:
            target_mask = (targets['damage_mask'] > 0.5).long()
            losses['seg'] = self.seg_loss(outputs['segmentation'], target_mask)
            losses['dice'] = self.dice_loss(outputs['segmentation'], target_mask)
            total_loss += losses['seg'] + 0.5 * losses['dice']
        
        # Multi-label loss
        if 'multilabel' in outputs:
            losses['ml'] = self.ml_loss(outputs['multilabel'], targets['multilabel'])
            total_loss += 2.0 * losses['ml']
        
        return total_loss, losses


class DiceLoss(nn.Module):
    """Dice loss"""
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)[:, 1]  # damage class
        intersection = (pred * target).sum()
        dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
        return 1 - dice


def get_param_groups(model, lr, model_type):
    """파라미터 그룹 설정"""
    if model_type == 'mask2former':
        # Mask2Former는 다른 learning rate
        return [
            {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            {'params': model.damage_head.pixel_decoder.parameters(), 'lr': lr * 0.5},
            {'params': model.damage_head.transformer_decoder.parameters(), 'lr': lr},
            {'params': model.damage_head.query_embed.parameters(), 'lr': lr}
        ]
    else:
        return [
            {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            {'params': model.damage_head.parameters(), 'lr': lr}
        ]


def get_scheduler(optimizer, epochs, steps_per_epoch):
    """스케줄러 생성"""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g['lr'] for g in optimizer.param_groups],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1
    )


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, score, model_type):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'score': score,
        'model_type': model_type
    }
    
    filename = f'best_{model_type}_damage_model.pth'
    torch.save(checkpoint, filename)


def print_epoch_summary(epoch, total_epochs, train_metrics, val_metrics):
    """에폭 요약 출력"""
    print(f"\nEpoch {epoch+1}/{total_epochs}:")
    print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['avg_f1']:.4f}")
    print(f"  Valid - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['avg_f1']:.4f}")
    
    if 'query_acc' in val_metrics:
        print(f"  Query Accuracy: {val_metrics['query_acc']:.4f}")


def get_score(metrics, model_type):
    """평가 점수 계산"""
    if model_type == 'mask2former':
        # Query accuracy도 고려
        return metrics['avg_f1'] * 0.7 + metrics.get('query_acc', 0) * 0.3
    else:
        return metrics['avg_f1']


def move_batch_to_device(batch, device):
    """배치를 디바이스로 이동"""
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }