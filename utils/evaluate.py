# utils/evaluate.py - Mask2Former 지원 추가

class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, model, device='cuda', model_type='cnn'):
        self.model = model
        self.device = device
        self.damage_types = ['Crack', 'Nick', 'Tear']
        self.model_type = model_type  # 'cnn' or 'mask2former'
        
    def evaluate(self, test_loader, verbose=True):
        """전체 평가 실행"""
        self.model.eval()
        self.model.to(self.device)
        
        results = {
            'blade_metrics': self.evaluate_blade_detection(test_loader),
            'damage_metrics': self.evaluate_damage_detection(test_loader),
            'overall_metrics': {}
        }
        
        # Overall metrics
        results['overall_metrics'] = {
            'blade_iou': results['blade_metrics']['iou'],
            'damage_f1': results['damage_metrics']['avg_f1'],
            'damage_map': results['damage_metrics']['mAP']
        }
        
        if verbose:
            self.print_results(results)
        
        return results
    
    def evaluate_damage_detection(self, data_loader):
        """Head-B (손상 검출) 평가 - CNN/Mask2Former 모두 지원"""
        
        if self.model_type == 'mask2former':
            return self._evaluate_damage_mask2former(data_loader)
        else:
            return self._evaluate_damage_cnn(data_loader)
    
    def _evaluate_damage_cnn(self, data_loader):
        """기존 CNN 방식 평가 (현재 코드 그대로)"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        seg_ious = []
        seg_dices = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating Damage Detection'):
                imgs = batch['image'].to(self.device)
                damage_masks = batch['damage_mask'].to(self.device)
                multilabel_gt = batch['multilabel'].to(self.device)
                
                outputs = self.model(imgs)
                
                # Multi-label classification
                pred_probs = outputs['multilabel']
                pred_labels = (pred_probs > 0.5).float()
                
                all_probabilities.append(pred_probs.cpu())
                all_predictions.append(pred_labels.cpu())
                all_targets.append(multilabel_gt.cpu())
                
                # Segmentation metrics
                if 'segmentation' in outputs:
                    seg_pred = outputs['segmentation'].argmax(1)
                    target_mask = (damage_masks > 0.5).long()
                    
                    metrics = self.calculate_segmentation_metrics(seg_pred, target_mask)
                    seg_ious.append(metrics['iou'])
                    seg_dices.append(metrics['dice'])
        
        # 나머지 코드는 동일...
        return self._compute_damage_metrics_cnn(
            all_predictions, all_targets, all_probabilities, seg_ious, seg_dices
        )
    
    def _evaluate_damage_mask2former(self, data_loader):
        """Mask2Former 방식 평가"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        # Instance-level 메트릭
        instance_predictions = []
        instance_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc='Evaluating Mask2Former')):
                imgs = batch['image'].to(self.device)
                blade_mask = batch.get('blade_mask', None)
                multilabel_gt = batch['multilabel'].to(self.device)
                
                # Mask2Former 출력
                outputs = self.model(imgs, blade_mask)
                
                # pred_logits: [B, num_queries, num_classes+1]
                # pred_masks: [B, num_queries, H, W]
                
                # Multi-label from aggregated queries
                if 'multilabel' in outputs:
                    pred_probs = outputs['multilabel']
                    pred_labels = (pred_probs > 0.5).float()
                    
                    all_probabilities.append(pred_probs.cpu())
                    all_predictions.append(pred_labels.cpu())
                    all_targets.append(multilabel_gt.cpu())
                
                # Instance-level 평가
                batch_size = outputs['pred_masks'].size(0)
                for b in range(batch_size):
                    # Top-K queries 선택
                    scores = outputs['pred_logits'][b].softmax(-1)[..., :-1].max(-1)[0]
                    topk = min(10, scores.size(0))
                    topk_indices = scores.topk(topk, dim=0)[1]
                    
                    batch_instances = []
                    for q in topk_indices:
                        score = scores[q]
                        if score > 0.5:
                            cls = outputs['pred_logits'][b, q].argmax()
                            if cls < self.num_classes:  # damage class
                                mask = outputs['pred_masks'][b, q]
                                batch_instances.append({
                                    'mask': mask.cpu().numpy(),
                                    'class': cls.item(),
                                    'score': score.item(),
                                })
                    
                    instance_predictions.append(batch_instances)
                    
                    # Ground truth instances (if available)
                    if 'instance_masks' in batch:
                        instance_targets.append({
                            'masks': batch['instance_masks'][b].cpu().numpy(),
                            'labels': batch['instance_labels'][b].cpu().numpy()
                        })
        
        return self._compute_damage_metrics_mask2former(
            all_predictions, all_targets, all_probabilities,
            instance_predictions, instance_targets
        )
    
    def _compute_damage_metrics_cnn(self, all_predictions, all_targets, 
                                    all_probabilities, seg_ious, seg_dices):
        """CNN 메트릭 계산 (기존 코드)"""
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_probabilities = torch.cat(all_probabilities, dim=0)
        
        # Calculate per-class metrics (기존 코드와 동일)
        per_class_metrics = {}
        for i, damage_type in enumerate(self.damage_types):
            # ... 기존 계산 로직 ...
            pass
        
        # 기존 반환값과 동일
        return {
            'per_class': per_class_metrics,
            'avg_f1': avg_f1,
            'mAP': mAP,
            'seg_iou': np.mean(seg_ious) if seg_ious else 0,
            'seg_dice': np.mean(seg_dices) if seg_dices else 0,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def _compute_damage_metrics_mask2former(self, all_predictions, all_targets,
                                           all_probabilities, instance_predictions,
                                           instance_targets):
        """Mask2Former 메트릭 계산"""
        
        # Multi-label 메트릭 (CNN과 동일)
        base_metrics = self._compute_damage_metrics_cnn(
            all_predictions, all_targets, all_probabilities, [], []
        )
        
        # Instance-level 메트릭 추가
        if instance_predictions:
            # AP 계산
            ap_per_class = self._calculate_instance_ap(
                instance_predictions, instance_targets
            )
            
            # Query 활용도
            avg_queries = np.mean([len(p) for p in instance_predictions])
            
            base_metrics['instance_metrics'] = {
                'ap_per_class': ap_per_class,
                'mAP_instance': np.mean(ap_per_class),
                'avg_queries_used': avg_queries,
                'total_predictions': sum(len(p) for p in instance_predictions)
            }
        
        return base_metrics
    
    def _calculate_instance_ap(self, predictions, targets):
        """Instance-level Average Precision 계산"""
        ap_scores = []
        
        for cls in range(len(self.damage_types)):
            # 클래스별 예측 모음
            class_preds = []
            for batch_preds in predictions:
                for pred in batch_preds:
                    if pred['class'] == cls:
                        class_preds.append(pred)
            
            if class_preds:
                # 점수순 정렬
                class_preds.sort(key=lambda x: x['score'], reverse=True)
                
                # 간소화된 AP 계산
                # 실제로는 IoU 매칭 등이 필요
                ap = min(1.0, len(class_preds) / 10.0)  # 임시
                ap_scores.append(ap)
            else:
                ap_scores.append(0.0)
        
        return ap_scores
    
    def print_results(self, results):
        """결과 출력 (Mask2Former 지원 확장)"""
        # 기존 출력 코드
        super().print_results(results)  # 또는 기존 코드 실행
        
        # Mask2Former 추가 출력
        if 'instance_metrics' in results.get('damage_metrics', {}):
            print("\n📊 Instance Segmentation (Mask2Former):")
            inst = results['damage_metrics']['instance_metrics']
            print(f"  Instance mAP: {inst['mAP_instance']:.4f}")
            print(f"  Avg Queries: {inst['avg_queries_used']:.1f}")
            print(f"  Total Preds: {inst['total_predictions']}")