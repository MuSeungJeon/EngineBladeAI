import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import json

# pycocotools가 필요합니다. pip install pycocotools
from pycocotools import mask as mask_utils


class FinalBladeDataset(Dataset):
    """
    최종 통합된 데이터셋(final_dataset)을 위한 Dataset 클래스.
    하나의 통합된 JSON 파일을 읽어 End-to-End 학습에 필요한 데이터를 생성합니다.
    """
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / self.split / 'images'
        
        json_path = self.root / self.split / 'annotations.json'
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.images_info = self.data['images']
        
        # 빠른 조회를 위해 image_id를 key로 하는 annotation 맵 생성
        self.annotations_map = {}
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_map:
                self.annotations_map[img_id] = []
            self.annotations_map[img_id].append(ann)
            
        # 기본 이미지 변환기
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        img_id = img_info['id']
        img_path = self.images_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # 원본 이미지 크기 저장
        original_w, original_h = image.size
        
        # 모델에 전달할 target 딕셔너리 생성
        target = {}
        
        # blade_mask 생성 (category_id == 1)
        blade_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        # Mask2Former가 사용할 손상 관련 정보
        damage_masks_np = []
        damage_labels = []
        
        annotations = self.annotations_map.get(img_id, [])
        for ann in annotations:
            cat_id = ann['category_id']
            rle = mask_utils.frPyObjects(ann['segmentation'], original_h, original_w)
            mask = mask_utils.decode(rle)
            
            if cat_id == 1: # 블레이드
                blade_mask = np.maximum(blade_mask, mask)
            else: # 손상
                damage_masks_np.append(mask)
                damage_labels.append(cat_id - 2) # 2,3,4 -> 0,1,2

        # 이미지와 마스크 변환 적용
        # (주의: Albumentations를 사용하면 더 효율적으로 마스크도 함께 변환 가능)
        image = self.transform(image)
        
        # blade_mask는 텐서로 변환 후 리사이즈
        blade_mask = torch.from_numpy(blade_mask).unsqueeze(0).float()
        blade_mask = F.interpolate(blade_mask.unsqueeze(0), size=(640, 640), mode='nearest').squeeze()
        
        # target 최종 구성
        target['blade_mask'] = blade_mask.long()
        target['labels'] = torch.tensor(damage_labels, dtype=torch.int64)
        
        # 손상 마스크도 리사이즈
        if damage_masks_np:
            damage_masks_tensor = torch.from_numpy(np.stack(damage_masks_np)).unsqueeze(1).float()
            resized_damage_masks = F.interpolate(damage_masks_tensor, size=(640, 640), mode='nearest').squeeze(1)
            target['masks'] = resized_damage_masks
        else: # 손상이 없는 경우
            target['masks'] = torch.zeros((0, 640, 640), dtype=torch.float32)

        return image, target