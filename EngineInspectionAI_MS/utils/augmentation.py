# utils/augmentation.py
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import random
import json
from typing import List, Dict, Tuple, Optional
from pycocotools import mask as maskUtils
import copy

class MultiLabelAugmentation:
    """멀티라벨 손상 데이터 증강 (좌표 변환 포함)"""
    
    def __init__(self, 
                 rotation_90_prob=0.5,
                 blur_prob=0.5,
                 brightness_prob=0.5,
                 blur_range=(0, 5),
                 brightness_range=(-0.4, 0.4)):
        self.rotation_90_prob = rotation_90_prob
        self.blur_prob = blur_prob
        self.brightness_prob = brightness_prob
        self.blur_range = blur_range
        self.brightness_range = brightness_range
    
    def rotate_90_coords(self, image, annotations, direction='random'):
        """
        90도 회전 with 좌표 변환
        Args:
            image: PIL Image
            annotations: COCO 형식 annotations 리스트
            direction: 'cw', 'ccw', 'flip', 'random'
        """
        if direction == 'random':
            direction = random.choice(['cw', 'ccw', 'flip'])
        
        # PIL Image로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        orig_w, orig_h = image.size
        
        # 이미지 회전
        if direction == 'cw':  # 시계방향 90도
            image = image.rotate(-90, expand=True)
            new_w, new_h = image.size
        elif direction == 'ccw':  # 반시계방향 90도
            image = image.rotate(90, expand=True)
            new_w, new_h = image.size
        elif direction == 'flip':  # 180도
            image = image.rotate(180, expand=True)
            new_w, new_h = orig_w, orig_h
        else:
            return image, annotations
        
        # Annotations 변환
        new_annotations = []
        for ann in annotations:
            new_ann = copy.deepcopy(ann)
            
            # 1. Segmentation 좌표 변환
            if 'segmentation' in new_ann and new_ann['segmentation']:
                new_segments = []
                for segment in new_ann['segmentation']:
                    new_segment = []
                    
                    # 좌표 쌍으로 처리
                    for i in range(0, len(segment), 2):
                        x, y = segment[i], segment[i+1]
                        
                        if direction == 'cw':  # 시계방향 90도
                            new_x = orig_h - y
                            new_y = x
                        elif direction == 'ccw':  # 반시계방향 90도
                            new_x = y
                            new_y = orig_w - x
                        elif direction == 'flip':  # 180도
                            new_x = orig_w - x
                            new_y = orig_h - y
                        
                        new_segment.extend([new_x, new_y])
                    
                    new_segments.append(new_segment)
                
                new_ann['segmentation'] = new_segments
            
            # 2. Bounding Box 변환
            if 'bbox' in new_ann:
                x, y, w, h = new_ann['bbox']
                
                if direction == 'cw':  # 시계방향 90도
                    new_x = orig_h - y - h
                    new_y = x
                    new_w = h
                    new_h = w
                elif direction == 'ccw':  # 반시계방향 90도
                    new_x = y
                    new_y = orig_w - x - w
                    new_w = h
                    new_h = w
                elif direction == 'flip':  # 180도
                    new_x = orig_w - x - w
                    new_y = orig_h - y - h
                    new_w = w
                    new_h = h
                
                new_ann['bbox'] = [new_x, new_y, new_w, new_h]
            
            # 3. Area 재계산
            if 'bbox' in new_ann:
                new_ann['area'] = new_ann['bbox'][2] * new_ann['bbox'][3]
            
            # 4. 멀티라벨은 유지
            # new_ann['multilabel'] 은 그대로 유지
            
            new_annotations.append(new_ann)
        
        return image, new_annotations
    
    def apply_blur(self, image, pixels=None):
        """블러 적용 (좌표는 변경 없음)"""
        if pixels is None:
            pixels = random.uniform(self.blur_range[0], self.blur_range[1])
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if pixels > 0:
            kernel_size = int(pixels * 2) + 1
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), pixels)
        
        return Image.fromarray(image)
    
    def adjust_brightness(self, image, factor=None):
        """밝기 조정 (좌표는 변경 없음)"""
        if factor is None:
            factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        brightness_factor = 1.0 + factor
        image = TF.adjust_brightness(image, brightness_factor)
        
        return image
    
    def __call__(self, image, annotations):
        """
        모든 증강 적용
        Args:
            image: PIL Image
            annotations: COCO 형식 annotations 리스트
        Returns:
            augmented_image, augmented_annotations
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        aug_annotations = copy.deepcopy(annotations)
        
        # 90도 회전 (좌표 변환 필요)
        if random.random() < self.rotation_90_prob:
            image, aug_annotations = self.rotate_90_coords(image, aug_annotations)
        
        # 블러 (좌표 변환 불필요)
        if random.random() < self.blur_prob:
            image = self.apply_blur(image)
        
        # 밝기 조정 (좌표 변환 불필요)
        if random.random() < self.brightness_prob:
            image = self.adjust_brightness(image)
        
        return image, aug_annotations


class COCOAugmentation:
    """COCO JSON 파일 전체 증강"""
    
    @staticmethod
    def augment_coco_json(input_json, output_json, augmentation_factor=2):
        """
        COCO JSON 파일 증강
        Args:
            input_json: 원본 damage_multilabel.json
            output_json: 증강된 결과 저장 경로
            augmentation_factor: 증강 배수 (2 = 2배로 증가)
        """
        with open(input_json, 'r') as f:
            data = json.load(f)
        
        augmentor = MultiLabelAugmentation()
        
        new_data = {
            'info': data['info'],
            'licenses': data['licenses'],
            'categories': data['categories'],
            'images': [],
            'annotations': []
        }
        
        img_id_counter = 0
        ann_id_counter = 0
        
        # 이미지별로 그룹화
        img_to_anns = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # 각 이미지 처리
        for img_info in data['images']:
            img_id = img_info['id']
            img_path = img_info['file_name']
            
            # 원본 추가
            new_img_info = img_info.copy()
            new_img_info['id'] = img_id_counter
            new_data['images'].append(new_img_info)
            
            # 원본 annotations 추가
            if img_id in img_to_anns:
                for ann in img_to_anns[img_id]:
                    new_ann = ann.copy()
                    new_ann['id'] = ann_id_counter
                    new_ann['image_id'] = img_id_counter
                    new_data['annotations'].append(new_ann)
                    ann_id_counter += 1
            
            img_id_counter += 1
            
            # 증강 버전들 추가
            for aug_idx in range(augmentation_factor - 1):
                # 이미지 로드 (실제 구현시)
                # image = Image.open(img_path)
                # annotations = img_to_anns.get(img_id, [])
                
                # 증강 적용
                # aug_image, aug_annotations = augmentor(image, annotations)
                
                # 새 이미지 정보
                aug_img_info = img_info.copy()
                aug_img_info['id'] = img_id_counter
                aug_img_info['file_name'] = f"aug_{aug_idx}_{img_info['file_name']}"
                new_data['images'].append(aug_img_info)
                
                # 증강된 annotations 추가
                if img_id in img_to_anns:
                    # 여기서 실제 증강 적용
                    aug_annotations = img_to_anns[img_id]  # 실제로는 증강된 것
                    
                    for ann in aug_annotations:
                        new_ann = ann.copy()
                        new_ann['id'] = ann_id_counter
                        new_ann['image_id'] = img_id_counter
                        # multilabel은 유지됨
                        new_data['annotations'].append(new_ann)
                        ann_id_counter += 1
                
                img_id_counter += 1
        
        # 저장
        with open(output_json, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        print(f"증강 완료:")
        print(f"  원본 이미지: {len(data['images'])}")
        print(f"  증강 후 이미지: {len(new_data['images'])}")
        print(f"  원본 annotations: {len(data['annotations'])}")
        print(f"  증강 후 annotations: {len(new_data['annotations'])}")
        
        return new_data


def visualize_multilabel_augmentation(image_path, annotations):
    """멀티라벨 증강 시각화"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    image = Image.open(image_path)
    augmentor = MultiLabelAugmentation()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 원본
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    
    # Segmentation 그리기
    for ann in annotations:
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape(-1, 2)
                polygon = Polygon(poly, fill=False, edgecolor='red', linewidth=2)
                axes[0, 0].add_patch(polygon)
        
        # 멀티라벨 표시
        if 'multilabel' in ann:
            labels = ann['multilabel']
            damage_types = ['Crack', 'Nick', 'Tear']
            active_labels = [damage_types[i] for i, v in enumerate(labels) if v]
            if active_labels:
                axes[0, 0].text(10, 30, ', '.join(active_labels), 
                              bbox=dict(boxstyle="round", facecolor='wheat'))
    
    axes[0, 0].axis('off')
    
    # 회전 증강들
    directions = ['cw', 'ccw', 'flip']
    titles = ['Rotate CW 90°', 'Rotate CCW 90°', 'Rotate 180°']
    
    for idx, (direction, title) in enumerate(zip(directions, titles)):
        aug_img, aug_anns = augmentor.rotate_90_coords(image, annotations, direction)
        axes[0, idx+1].imshow(aug_img)
        axes[0, idx+1].set_title(title)
        
        # 변환된 segmentation 그리기
        for ann in aug_anns:
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2)
                    polygon = Polygon(poly, fill=False, edgecolor='blue', linewidth=2)
                    axes[0, idx+1].add_patch(polygon)
        
        axes[0, idx+1].axis('off')
    
    # 블러와 밝기
    blur_img = augmentor.apply_blur(image, 3)
    axes[1, 0].imshow(blur_img)
    axes[1, 0].set_title('Blur 3px')
    axes[1, 0].axis('off')
    
    bright_img = augmentor.adjust_brightness(image, 0.3)
    axes[1, 1].imshow(bright_img)
    axes[1, 1].set_title('Bright 30%')
    axes[1, 1].axis('off')
    
    dark_img = augmentor.adjust_brightness(image, -0.3)
    axes[1, 2].imshow(dark_img)
    axes[1, 2].set_title('Dark 30%')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()