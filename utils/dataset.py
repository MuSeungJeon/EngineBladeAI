# utils/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path
import json
from PIL import Image
import numpy as np
from pycocotools import mask as maskUtils
from torchvision import transforms


class UnifiedDamageDataset(Dataset):
    """통합 손상 검출 데이터셋"""
    
    def __init__(
        self,
        blade_data_root=None,  # 블레이드 데이터 경로
        damage_data_root=None,  # 손상 데이터 경로
        split='train',
        transform=None,
        target_size=(640, 640),
        use_mask2former=False  # 추가
    ):
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.use_mask2former = use_mask2former  # 추가
        
         # 블레이드 데이터 로드 (Head-A)
        if blade_data_root:
            blade_json = Path(blade_data_root) / split / 'blade_only.json'
            with open(blade_json, 'r') as f:
                self.blade_data = json.load(f)
            self.blade_root = Path(blade_data_root) / split
        else:
            self.blade_data = None
            
        # 손상 데이터 로드 (Head-B)
        if damage_data_root:
            damage_json = Path(damage_data_root) / split / 'damage_multilabel.json'
            with open(damage_json, 'r') as f:
                self.damage_data = json.load(f)
            self.damage_root = Path(damage_data_root) / split
        else:
            self.damage_data = None
        
        # 이미지 매칭 (파일명 기준)
        self.matched_samples = self._match_samples()
        
    def _match_samples(self):
        """블레이드와 손상 데이터 매칭"""
        matched = []
        
        # 블레이드 이미지 인덱싱
        blade_imgs = {}
        if self.blade_data:
            for img in self.blade_data['images']:
                blade_imgs[img['file_name']] = img
        
        # 손상 이미지와 매칭
        if self.damage_data:
            for img in self.damage_data['images']:
                fname = img['file_name']
                sample = {
                    'damage_img': img,
                    'blade_img': blade_imgs.get(fname),
                    'file_name': fname
                }
                matched.append(sample)
        
        return matched
    
    def __len__(self):
        return len(self.matched_samples)
    
    def __getitem__(self, idx):
        sample = self.matched_samples[idx]
        
        # 이미지 로드 (손상 데이터 경로 우선)
        img_path = self.damage_root / 'images' / sample['file_name']
        if not img_path.exists():
            img_path = self.damage_root / sample['file_name']
        
        image = Image.open(img_path).convert('RGB')
        
        # 블레이드 마스크 생성
        blade_mask = np.zeros(self.target_size, dtype=np.uint8)
        if sample['blade_img'] and self.blade_data:
            # 블레이드 annotation 찾기
            img_id = sample['blade_img']['id']
            for ann in self.blade_data['annotations']:
                if ann['image_id'] == img_id:
                    # 마스크 생성 로직
                    blade_mask = self._create_mask_from_annotation(ann, sample['blade_img'])
                    break
        
        # 손상 정보 추출
        damage_mask = np.zeros(self.target_size, dtype=np.uint8)
        multilabel = np.zeros(3, dtype=np.float32)
        instance_masks = []
        instance_labels = []
        
        if sample['damage_img'] and self.damage_data:
            img_id = sample['damage_img']['id']
            for ann in self.damage_data['annotations']:
                if ann['image_id'] == img_id:
                    # Multi-label 추출
                    if 'multilabel' in ann:
                        for i, label in enumerate(ann['multilabel']):
                            if label:
                                multilabel[i] = 1.0
                                
                    # Mask2Former용 instance 추가
                    if self.use_mask2former and 'segmentation' in ann:
                        mask = self._create_mask_from_annotation(ann, sample['damage_img'])
                        for i, label in enumerate(ann.get('multilabel', [])):
                            if label:
                                instance_masks.append(torch.tensor(mask))
                                instance_labels.append(i)
        
        # Transform 적용
        if self.transform:
            if hasattr(self.transform, '__module__') and 'albumentations' in self.transform.__module__:
                transformed = self.transform(
                    image=np.array(image),
                    masks=[blade_mask, damage_mask]
                )
                image = transformed['image']
                blade_mask, damage_mask = transformed['masks']
            else:
                image = self.transform(image)
                blade_mask = torch.tensor(blade_mask, dtype=torch.long)
                damage_mask = torch.tensor(damage_mask, dtype=torch.float32)
        else:
            # transform이 없을 때만 ToTensor 적용
            image = transforms.ToTensor()(image)
            blade_mask = torch.tensor(blade_mask, dtype=torch.long)
            damage_mask = torch.tensor(damage_mask, dtype=torch.float32)
        
        # 이미 텐서인 경우 다시 변환하지 않음
        if not torch.is_tensor(blade_mask):
            blade_mask = torch.tensor(blade_mask, dtype=torch.long)
        if not torch.is_tensor(multilabel):
            multilabel = torch.tensor(multilabel, dtype=torch.float32)
        
        # Mask2Former용 타겟 포맷
        if self.use_mask2former:
            return {
                'image': image,
                'blade_mask': blade_mask,
                'instance_masks': torch.stack(instance_masks) if instance_masks else torch.zeros(1, *self.target_size),
                'instance_labels': torch.tensor(instance_labels) if instance_labels else torch.zeros(1, dtype=torch.long),
                'multilabel': multilabel
            }
        
        return {
            'image': image,
            'blade_mask': blade_mask,
            'damage_mask': torch.tensor(damage_mask),
            'multilabel': multilabel
        }   
    
    def _create_blade_mask(self, annotations, height, width):
        """블레이드 마스크 생성"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in annotations:
            if ann.get('category_id') == 1:  # blade category
                if 'segmentation' in ann:
                    # COCO polygon to mask
                    from pycocotools import mask as maskUtils
                    rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
                    m = maskUtils.decode(rle)
                    if len(m.shape) == 3:
                        m = m.sum(axis=2)
                    mask[m > 0] = 1
        
        return mask
    
    def _create_damage_mask(self, annotations, height, width):
        """손상 마스크 생성"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in annotations:
            if ann.get('category_id') > 1:  # damage categories
                if 'segmentation' in ann:
                    from pycocotools import mask as maskUtils
                    rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
                    m = maskUtils.decode(rle)
                    if len(m.shape) == 3:
                        m = m.sum(axis=2)
                    mask[m > 0] = ann['category_id'] - 1  # 1-indexed to 0-indexed
        
        return mask
    
    def _create_multilabel(self, annotations):
        """Multi-label 벡터 생성"""
        multilabel = np.zeros(3, dtype=np.float32)  # [crack, nick, tear]
        
        for ann in annotations:
            if 'multilabel' in ann:
                for i, label in enumerate(ann['multilabel']):
                    if label:
                        multilabel[i] = 1.0
        
        return multilabel
    
    def _ann_to_mask(self, annotation, height, width):
        """단일 annotation을 마스크로 변환 (Mask2Former용)"""
        mask = np.zeros((height, width), dtype=np.float32)
        
        if 'segmentation' in annotation:
            from pycocotools import mask as maskUtils
            rle = maskUtils.frPyObjects(annotation['segmentation'], height, width)
            m = maskUtils.decode(rle)
            if len(m.shape) == 3:
                m = m.sum(axis=2)
            mask[m > 0] = 1.0
        
        return torch.tensor(mask, dtype=torch.float32)
    
    def _create_mask_from_annotation(self, annotation, img_info):
        """annotation을 마스크로 변환"""
        from pycocotools import mask as maskUtils
        
        height = img_info['height']
        width = img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if 'segmentation' in annotation:
            if isinstance(annotation['segmentation'], list):
                # Polygon format
                rle = maskUtils.frPyObjects(annotation['segmentation'], height, width)
                m = maskUtils.decode(rle)
                if len(m.shape) == 3:
                    m = m.sum(axis=2)
                mask[m > 0] = 1
            elif isinstance(annotation['segmentation'], dict):
                # RLE format
                m = maskUtils.decode(annotation['segmentation'])
                mask[m > 0] = 1
        elif 'bbox' in annotation:
            # Bounding box fallback
            x, y, w, h = annotation['bbox']
            mask[int(y):int(y+h), int(x):int(x+w)] = 1
            
        return mask


def custom_collate_fn(batch):
    """배치 데이터 정리"""
    images = torch.stack([item['image'] for item in batch])
    blade_masks = torch.stack([item['blade_mask'] for item in batch])
    damage_masks = torch.stack([item['damage_mask'] for item in batch])
    multilabels = torch.stack([item['multilabel'] for item in batch])
    file_names = [item['file_name'] for item in batch]
    
    return {
        'image': images,
        'blade_mask': blade_masks,
        'damage_mask': damage_masks,
        'multilabel': multilabels,
        'file_name': file_names
    }



def mask2former_collate_fn(batch):
    """Mask2Former용 커스텀 collate function
    Instance masks와 labels가 배치마다 다른 수일 수 있으므로 특별 처리
    """
    import torch
    
    # 기본 텐서들 (동일한 크기)
    images = torch.stack([b['image'] for b in batch])
    blade_masks = torch.stack([b['blade_mask'] for b in batch])
    multilabels = torch.stack([b['multilabel'] for b in batch])
    
    # Instance masks와 labels는 리스트로 유지
    # 각 배치 아이템이 다른 수의 인스턴스를 가질 수 있음
    instance_masks = []
    instance_labels = []
    
    for b in batch:
        if 'instance_masks' in b:
            instance_masks.append(b['instance_masks'])
            instance_labels.append(b['instance_labels'])
        else:
            # 빈 인스턴스 추가
            instance_masks.append(torch.zeros(1, 640, 640))
            instance_labels.append(torch.zeros(1, dtype=torch.long))
    
    # damage_mask가 있으면 포함 (선택적)
    damage_masks = None
    if 'damage_mask' in batch[0]:
        damage_masks = torch.stack([b['damage_mask'] for b in batch])
    
    # img_info도 포함 (디버깅용)
    img_infos = [b.get('img_info', {}) for b in batch] if 'img_info' in batch[0] else None
    
    collated = {
        'image': images,
        'blade_mask': blade_masks,
        'multilabel': multilabels,
        'instance_masks': instance_masks,  # 리스트
        'instance_labels': instance_labels,  # 리스트
    }
    
    # 선택적 필드들
    if damage_masks is not None:
        collated['damage_mask'] = damage_masks
    if img_infos:
        collated['img_info'] = img_infos
    
    return collated


def create_dataloaders(
    blade_data_root=None,  # 추가
    damage_data_root=None,  # 추가  
    batch_size=4,
    num_workers=4,
    model_type='cnn'
):
    """데이터로더 생성 (Mask2Former 지원 포함)
    
    Args:
        data_root: 데이터 루트 경로
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        model_type: 'cnn' or 'mask2former'
    
    Returns:
        train_loader, valid_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # Mask2Former 여부 확인
    use_mask2former = (model_type == 'mask2former')
    
    # 데이터셋 생성
    train_dataset = UnifiedDamageDataset(
        blade_data_root=blade_data_root,
        damage_data_root=damage_data_root,
        split='train',
        transform=get_train_transforms(),
        use_mask2former=use_mask2former
    )
    
    valid_dataset = UnifiedDamageDataset(
        blade_data_root=blade_data_root,
        damage_data_root=damage_data_root,
        split='valid',
        transform=get_valid_transforms(),
        use_mask2former=use_mask2former
    )
    
    test_dataset = UnifiedDamageDataset(
        blade_data_root=blade_data_root,
        damage_data_root=damage_data_root,
        split='test',
        transform=get_valid_transforms(),
        use_mask2former=use_mask2former
    )
    
    # Collate function 선택
    collate_fn = mask2former_collate_fn if use_mask2former else None
    
    # 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True  # Mask2Former는 배치 크기 일정하게 유지
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False  # Validation은 drop_last 불필요
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False  # Test도 drop_last 불필요
    )
    
    return train_loader, valid_loader, test_loader

def get_train_transforms(img_size=640):
    """학습용 변환"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_valid_transforms(img_size=640):
    """검증/테스트용 변환"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])