# import json
# import cv2
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# from pycocotools import mask as maskUtils

# def keep_blade_only(input_path, output_path):
#     """COCO JSON에서 Blade 라벨만 남기기"""
    
#     # JSON 로드
#     with open(input_path, 'r') as f:
#         data = json.load(f)
    
#     # 원본 정보 출력
#     print(f"\n📁 파일: {input_path.name}")
#     print(f"원본 카테고리: {[cat['name'] for cat in data['categories']]}")
#     print(f"전체 annotations: {len(data['annotations'])}")
    
#     # Blade 카테고리 ID 찾기
#     blade_id = None
#     for cat in data['categories']:
#         if cat['name'].lower() == 'blade':
#             blade_id = cat['id']
#             break
    
#     if blade_id is None:
#         print("❌ Blade 카테고리 없음!")
#         return None
    
#     # Blade annotation만 필터링
#     blade_anns = []
#     for ann in data['annotations']:
#         if ann['category_id'] == blade_id:
#             ann['category_id'] = 1  # ID 통일
#             blade_anns.append(ann)
    
#     print(f"✅ Blade annotations: {len(blade_anns)}")
    
#     # 새 데이터 생성
#     new_data = {
#         'info': data['info'],
#         'licenses': data['licenses'],
#         'categories': [{'id': 1, 'name': 'Blade', 'supercategory': 'objects'}],
#         'images': data['images'],
#         'annotations': blade_anns
#     }
    
#     # 저장
#     with open(output_path, 'w') as f:
#         json.dump(new_data, f, indent=2)
    
#     return new_data

# def yolo_to_coco_blade_only(yolo_dir, output_json_path):
#     """YOLOv8 형식을 COCO JSON으로 변환 (Blade만)"""
    
#     yolo_dir = Path(yolo_dir)
    
#     # COCO 구조 초기화
#     coco_data = {
#         'info': {'description': 'Blade segmentation from YOLOv8'},
#         'licenses': [{'id': 1, 'name': 'CC BY 4.0'}],
#         'categories': [{'id': 1, 'name': 'Blade', 'supercategory': 'objects'}],
#         'images': [],
#         'annotations': []
#     }
    
#     ann_id = 0
    
#     # 각 split 처리
#     for split in ['train', 'valid', 'test']:
#         split_dir = yolo_dir / split
#         if not split_dir.exists():
#             continue
            
#         images_dir = split_dir / 'images'
#         labels_dir = split_dir / 'labels'
        
#         # 이미지별 처리
#         for img_id, img_path in enumerate(images_dir.glob('*.jpg')):
#             # 이미지 정보
#             img = cv2.imread(str(img_path))
#             height, width = img.shape[:2]
            
#             coco_data['images'].append({
#                 'id': img_id,
#                 'file_name': img_path.name,
#                 'height': height,
#                 'width': width
#             })
            
#             # 라벨 파일
#             label_path = labels_dir / f"{img_path.stem}.txt"
#             if not label_path.exists():
#                 continue
            
#             # YOLO 라벨 읽기
#             with open(label_path, 'r') as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if not parts:
#                         continue
                    
#                     class_id = int(parts[0])
                    
#                     # Blade(class_id=0)만 처리
#                     if class_id != 0:
#                         continue
                    
#                     # 폴리곤 좌표 (정규화됨)
#                     coords = list(map(float, parts[1:]))
                    
#                     # 픽셀 좌표로 변환
#                     polygon = []
#                     for i in range(0, len(coords), 2):
#                         x = coords[i] * width
#                         y = coords[i+1] * height
#                         polygon.extend([x, y])
                    
#                     # 바운딩 박스 계산
#                     x_coords = polygon[::2]
#                     y_coords = polygon[1::2]
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
                    
#                     coco_data['annotations'].append({
#                         'id': ann_id,
#                         'image_id': img_id,
#                         'category_id': 1,  # Blade
#                         'segmentation': [polygon],
#                         'area': (x_max - x_min) * (y_max - y_min),
#                         'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
#                         'iscrowd': 0
#                     })
                    
#                     ann_id += 1
        
#         # 각 split별로 저장
#         output_path = yolo_dir / split / 'blade_only.json'
#         split_data = {
#             'info': coco_data['info'],
#             'licenses': coco_data['licenses'],
#             'categories': coco_data['categories'],
#             'images': [img for img in coco_data['images'] 
#                       if any(img_path.parent.name == split for img_path in 
#                             (yolo_dir / split / 'images').glob(img['file_name']))],
#             'annotations': [ann for ann in coco_data['annotations'] 
#                           if ann['image_id'] in [img['id'] for img in coco_data['images']]]
#         }
        
#         with open(output_path, 'w') as f:
#             json.dump(split_data, f, indent=2)
        
#         print(f"✅ {split}: {len(split_data['annotations'])} blade annotations saved")
    
#     return coco_data

# def extract_damage_labels(input_json, output_json):
#     """손상 라벨만 추출 (블레이드 제외)"""
#     with open(input_json, 'r') as f:
#         data = json.load(f)
    
#     damage_categories = []
#     damage_category_ids = []
    
#     for cat in data['categories']:
#         name = cat['name'].lower()
#         if name not in ['blade', 'blades', 'background', 'objects']:
#             damage_categories.append(cat)
#             damage_category_ids.append(cat['id'])
    
#     damage_annotations = [
#         ann for ann in data['annotations']
#         if ann['category_id'] in damage_category_ids
#     ]
    
#     damage_data = {
#         'info': data['info'],
#         'licenses': data['licenses'],
#         'categories': damage_categories,
#         'images': data['images'],
#         'annotations': damage_annotations
#     }
    
#     # 카테고리 ID 재매핑
#     id_mapping = {old_id: i for i, old_id in enumerate(damage_category_ids)}
    
#     for ann in damage_data['annotations']:
#         ann['category_id'] = id_mapping[ann['category_id']]
    
#     for i, cat in enumerate(damage_data['categories']):
#         cat['id'] = i
    
#     with open(output_json, 'w') as f:
#         json.dump(damage_data, f, indent=2)
    
#     print(f"손상 라벨 추출 완료:")
#     print(f"  카테고리: {[cat['name'] for cat in damage_categories]}")
#     print(f"  Annotations: {len(damage_annotations)}")
    
#     return damage_data


# def process_yolo_splits(yolo_root):
#     """각 split별로 처리"""
#     yolo_root = Path(yolo_root)
    
#     for split in ['train', 'valid', 'test']:
#         split_path = yolo_root / split
#         if not split_path.exists():
#             print(f"⚠️ {split} 폴더 없음")
#             continue
        
#         images_dir = split_path / 'images'
#         labels_dir = split_path / 'labels'
        
#         # COCO 데이터 생성
#         coco_data = {
#             'info': {'description': f'Blade segmentation - {split}'},
#             'licenses': [{'id': 1, 'name': 'CC BY 4.0'}],
#             'categories': [{'id': 1, 'name': 'Blade'}],
#             'images': [],
#             'annotations': []
#         }
        
#         ann_id = 0
        
#         # 이미지 처리
#         for img_id, img_path in enumerate(images_dir.glob('*.jpg')):
#             img = cv2.imread(str(img_path))
#             h, w = img.shape[:2]
            
#             coco_data['images'].append({
#                 'id': img_id,
#                 'file_name': img_path.name,
#                 'height': h,
#                 'width': w
#             })
            
#             # 라벨 처리
#             label_file = labels_dir / f"{img_path.stem}.txt"
#             if label_file.exists():
#                 with open(label_file, 'r') as f:
#                     for line in f:
#                         parts = line.strip().split()
#                         if int(parts[0]) == 0:  # Blade
#                             coords = list(map(float, parts[1:]))
#                             polygon = []
#                             for i in range(0, len(coords), 2):
#                                 polygon.extend([coords[i] * w, coords[i+1] * h])
                            
#                             x_coords = polygon[::2]
#                             y_coords = polygon[1::2]
#                             bbox = [min(x_coords), min(y_coords), 
#                                    max(x_coords) - min(x_coords), 
#                                    max(y_coords) - min(y_coords)]
                            
#                             coco_data['annotations'].append({
#                                 'id': ann_id,
#                                 'image_id': img_id,
#                                 'category_id': 1,
#                                 'segmentation': [polygon],
#                                 'bbox': bbox,
#                                 'area': bbox[2] * bbox[3],
#                                 'iscrowd': 0
#                             })
#                             ann_id += 1
        
#         # 저장
#         output_path = split_path / 'blade_only.json'
#         with open(output_path, 'w') as f:
#             json.dump(coco_data, f, indent=2)
        
#         print(f"{split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")


# class MultiLabelReviewer:
#     """멀티라벨 수동 검토 도구"""
    
#     def __init__(self, data_dir, json_file):
#         self.data_dir = Path(data_dir)
#         self.json_file = json_file
        
#         with open(json_file, 'r') as f:
#             self.data = json.load(f)
        
#         self.annotations = self.data['annotations']
#         self.images = {img['id']: img for img in self.data['images']}
#         self.categories = self.data['categories']
        
#         self.current_idx = 0
#         self.review_results = []
#         self.damage_types = ['Crack', 'Nick', 'Tear', 'Tipcurl']
        
#     def review_annotation(self, idx=None):
#         """특정 annotation 검토"""
#         if idx is not None:
#             self.current_idx = idx
            
#         if self.current_idx >= len(self.annotations):
#             print("모든 검토 완료!")
#             return None
        
#         ann = self.annotations[self.current_idx]
#         img_info = self.images[ann['image_id']]
        
#         img_path = self.data_dir / img_info['file_name']
#         img = Image.open(img_path)
#         img_array = np.array(img)
        
#         h, w = img_info['height'], img_info['width']
        
#         if 'segmentation' in ann and ann['segmentation']:
#             rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
#             mask = maskUtils.decode(rle)
#             if len(mask.shape) == 3:
#                 mask = mask.sum(axis=2)
#         else:
#             mask = np.zeros((h, w))
        
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
#         axes[0].imshow(img_array)
#         axes[0].set_title('Original Image')
#         axes[0].axis('off')
        
#         axes[1].imshow(mask, cmap='gray')
#         axes[1].set_title('Damage Mask')
#         axes[1].axis('off')
        
#         overlay = img_array.copy()
#         overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
#         axes[2].imshow(overlay.astype(np.uint8))
#         axes[2].set_title('Overlay')
#         axes[2].axis('off')
        
#         if 'bbox' in ann:
#             x, y, w, h = ann['bbox']
#             rect = patches.Rectangle((x, y), w, h, 
#                                     linewidth=2, edgecolor='yellow', 
#                                     facecolor='none')
#             axes[2].add_patch(rect)
        
#         plt.suptitle(f'Annotation {self.current_idx}/{len(self.annotations)}')
#         plt.show()
        
#         if 'multilabel' in ann:
#             current_labels = ann['multilabel']
#         elif 'category_id' in ann:
#             current_labels = [0] * len(self.damage_types)
#             if ann['category_id'] < len(current_labels):
#                 current_labels[ann['category_id']] = 1
#         else:
#             current_labels = [0] * len(self.damage_types)
        
#         print("\n현재 라벨:")
#         for i, damage_type in enumerate(self.damage_types):
#             status = "✓" if current_labels[i] else " "
#             print(f"[{status}] {i+1}. {damage_type}")
        
#         return ann, img_array, mask
    
#     def save_results(self, output_file):
#         """검토 결과 저장"""
#         reviewed_data = self.data.copy()
        
#         for result in self.review_results:
#             idx = result['idx']
#             multilabel = result['multilabel']
#             reviewed_data['annotations'][idx]['multilabel'] = multilabel
#             reviewed_data['annotations'][idx]['reviewed'] = True
        
#         with open(output_file, 'w') as f:
#             json.dump(reviewed_data, f, indent=2)
        
#         print(f"\n검토 결과 저장: {output_file}")
#         print(f"검토된 annotation: {len(self.review_results)}개")

# class BatchMultiLabelReviewer(MultiLabelReviewer):
#     """대량 검토를 위한 도구"""
    
#     def batch_review_with_breaks(self, batch_size=50, break_time=5):
#         """배치 단위로 검토"""
#         total = len(self.annotations)
#         batches = (total + batch_size - 1) // batch_size
        
#         print(f"총 {total}개 annotation을 {batches}개 배치로 검토")
#         print(f"배치당 {batch_size}개, 배치 간 {break_time}분 휴식\n")
        
#         for batch_num in range(batches):
#             start_idx = batch_num * batch_size
#             end_idx = min(start_idx + batch_size, total)
            
#             print(f"\n배치 {batch_num+1}/{batches} 시작 ({start_idx}-{end_idx})")
            
#             for idx in range(start_idx, end_idx):
#                 self.current_idx = idx
#                 self.review_annotation()
                
#                 user_input = input(f"[{idx+1}/{total}] 손상 번호 (1,3 또는 0=없음, s=건너뛰기): ")
                
#                 if user_input.lower() == 's':
#                     continue
#                 elif user_input == '0':
#                     multilabel = [0] * len(self.damage_types)
#                 else:
#                     multilabel = [0] * len(self.damage_types)
#                     for num in user_input.replace(',', ' ').split():
#                         try:
#                             multilabel[int(num)-1] = 1
#                         except:
#                             pass
                
#                 self.review_results.append({
#                     'idx': idx,
#                     'multilabel': multilabel
#                 })
            
#             self.save_checkpoint(f'checkpoint_batch_{batch_num+1}.json')
            
#             if batch_num < batches - 1:
#                 print(f"\n배치 {batch_num+1} 완료! 계속하려면 Enter")
#                 if input().lower() == 'q':
#                     break
                    
#         return self.review_results
    
#     def save_checkpoint(self, checkpoint_file):
#         """중간 저장"""
#         with open(checkpoint_file, 'w') as f:
#             json.dump(self.review_results, f)
#         print(f"체크포인트 저장: {checkpoint_file}")


    
# class LimitedBatchReviewer(BatchMultiLabelReviewer):
#     """출력 제한을 위한 개선된 검토 도구"""
    
#     def review_range(self, start_idx=0, max_count=250):
#         """특정 범위만 검토"""
#         total = len(self.annotations)
#         end_idx = min(start_idx + max_count, total)
        
#         print(f"검토 범위: {start_idx} ~ {end_idx-1} (총 {end_idx-start_idx}개)")
#         print(f"전체 진행도: {start_idx}/{total} ~ {end_idx}/{total}")
#         print("="*60)
#         print("입력: 숫자(1-5), 0=없음, s=건너뛰기, q=종료")
#         print("="*60)
        
#         for idx in range(start_idx, end_idx):
#             self.current_idx = idx
            
#             # 매 50개마다 진행상황 표시
#             if (idx - start_idx) % 50 == 0 and idx != start_idx:
#                 print(f"\n--- 진행: {idx-start_idx}/{max_count} 완료 ---\n")
            
#             self.review_annotation()
            
#             user_input = input(f"[{idx}/{total}] 입력: ").strip()
            
#             if user_input.lower() == 'q':
#                 print(f"종료. 마지막 인덱스: {idx}")
#                 break
#             elif user_input.lower() == 's':
#                 continue
#             elif user_input == '0':
#                 multilabel = [0] * 5
#             else:
#                 multilabel = [0] * 5
#                 for num in user_input.replace(',', ' ').split():
#                     try:
#                         n = int(num) - 1
#                         if 0 <= n < 5:
#                             multilabel[n] = 1
#                     except:
#                         pass
            
#             self.review_results.append({
#                 'idx': idx,
#                 'multilabel': multilabel
#             })
        
#         # 체크포인트 저장
#         checkpoint_name = f'checkpoint_{start_idx}_{end_idx}.json'
#         self.save_checkpoint(checkpoint_name)
        
#         print(f"\n완료: {len(self.review_results)}개 검토")
#         print(f"다음 시작 인덱스: {end_idx}")
        
#         return self.review_results, end_idx

# LabelingEditor.py
import json
import cv2
import yaml
from pathlib import Path
import numpy as np
from pycocotools import mask as maskUtils

def extract_blade_only(input_json, output_json):
    """COCO JSON에서 Blade 라벨만 추출"""
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"\n📁 파일: {Path(input_json).name}")
    print(f"원본 카테고리: {[cat['name'] for cat in data['categories']]}")
    print(f"전체 annotations: {len(data['annotations'])}")
    
    # Blade 카테고리 찾기
    blade_id = None
    for cat in data['categories']:
        if cat['name'] == 'Blade':
            blade_id = cat['id']
            break
    
    if blade_id is None:
        print("❌ Blade 카테고리 없음!")
        return None
    
    # Blade annotation만 필터링
    blade_anns = []
    for ann in data['annotations']:
        if ann['category_id'] == blade_id:
            new_ann = ann.copy()
            new_ann['category_id'] = 1  # 새 ID로 재매핑
            blade_anns.append(new_ann)
    
    print(f"✅ Blade annotations: {len(blade_anns)}")
    
    # 새 데이터 생성
    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': [{'id': 1, 'name': 'Blade', 'supercategory': 'object'}],
        'images': data['images'],
        'annotations': blade_anns
    }
    
    with open(output_json, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    return new_data

def extract_damage_only(input_json, output_json):
    """COCO JSON에서 손상 라벨만 추출 (Crack, Nick, Tear)"""
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"\n📁 파일: {Path(input_json).name}")
    print(f"원본 카테고리: {[(cat['id'], cat['name']) for cat in data['categories']]}")
    
    # 손상 카테고리만 필터링 (Crack, Nick, Tear)
    damage_categories = []
    damage_category_ids = []
    damage_names = ['Crack', 'Nick', 'Tear']  # 3개 손상만
    
    for cat in data['categories']:
        if cat['name'] in damage_names:
            damage_categories.append(cat)
            damage_category_ids.append(cat['id'])
    
    print(f"손상 카테고리: {[cat['name'] for cat in damage_categories]}")
    
    if not damage_categories:
        print("❌ 손상 카테고리 없음!")
        return None
    
    # 손상 annotation만 필터링
    damage_annotations = []
    for ann in data['annotations']:
        if ann['category_id'] in damage_category_ids:
            damage_annotations.append(ann.copy())
    
    # 카테고리 ID 재매핑 (0부터 시작)
    id_mapping = {old_id: i for i, old_id in enumerate(damage_category_ids)}
    
    # 새 카테고리 리스트
    new_categories = []
    for i, cat in enumerate(damage_categories):
        new_cat = cat.copy()
        new_cat['id'] = i
        new_cat['supercategory'] = 'damage'
        new_categories.append(new_cat)
    
    # Annotation의 category_id 업데이트
    for ann in damage_annotations:
        ann['category_id'] = id_mapping[ann['category_id']]
    
    # 새 데이터 생성
    damage_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': new_categories,
        'images': data['images'],
        'annotations': damage_annotations
    }
    
    with open(output_json, 'w') as f:
        json.dump(damage_data, f, indent=2)
    
    print(f"✅ 손상 annotations: {len(damage_annotations)}")
    
    return damage_data

def convert_yolo_to_coco(yolo_dir, output_json, yaml_file=None):
    """YOLO 형식을 COCO JSON으로 변환"""
    yolo_dir = Path(yolo_dir)
    images_dir = yolo_dir / 'images'
    labels_dir = yolo_dir / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ images 또는 labels 폴더가 없습니다: {yolo_dir}")
        return None
    
    # data.yaml에서 클래스 정보 읽기
    if yaml_file is None:
        yaml_file = yolo_dir.parent / 'data.yaml'
    
    class_names = ['Blade', 'Crack', 'Nick', 'Tear']  # YOLO 기본값
    
    if Path(yaml_file).exists():
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
            class_names = yaml_data.get('names', class_names)
            print(f"클래스 정보: {class_names}")
    else:
        print(f"⚠️ data.yaml 없음, 기본값 사용: {class_names}")
    
    # COCO 형식 초기화
    coco_data = {
        'info': {
            'description': f'Converted from YOLO - {yolo_dir.name}',
            'version': '1.0',
            'year': 2025
        },
        'licenses': [{'id': 1, 'name': 'CC BY 4.0'}],
        'categories': [],
        'images': [],
        'annotations': []
    }
    
    # 카테고리 생성 (YOLO는 0부터 시작)
    for i, name in enumerate(class_names):
        coco_data['categories'].append({
            'id': i,  # YOLO와 동일하게 0부터
            'name': name,
            'supercategory': 'object'
        })
    
    ann_id = 0
    img_id = 0
    
    # 이미지별 처리
    for img_path in sorted(images_dir.glob('*.jpg')):
        # 이미지 정보
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        coco_data['images'].append({
            'id': img_id,
            'file_name': img_path.name,
            'height': h,
            'width': w
        })
        
        # 라벨 파일 읽기
        label_file = labels_dir / f"{img_path.stem}.txt"
        if not label_file.exists():
            img_id += 1
            continue
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:  # 최소 class + 4개 좌표
                    continue
                
                class_id = int(parts[0])
                
                # YOLO class가 범위 내인지 확인
                if class_id >= len(class_names):
                    print(f"⚠️ 잘못된 class ID {class_id} (파일: {label_file.name})")
                    continue
                
                # YOLO 폴리곤 좌표 (정규화된 값)
                coords = list(map(float, parts[1:]))
                
                # COCO 폴리곤으로 변환 (픽셀 좌표)
                polygon = []
                for i in range(0, len(coords), 2):
                    if i+1 < len(coords):
                        x = coords[i] * w
                        y = coords[i+1] * h
                        polygon.extend([x, y])
                
                if len(polygon) < 6:  # 최소 3개 점
                    continue
                
                # 바운딩 박스 계산
                x_coords = polygon[::2]
                y_coords = polygon[1::2]
                x_min = min(x_coords)
                y_min = min(y_coords)
                bbox_w = max(x_coords) - x_min
                bbox_h = max(y_coords) - y_min
                
                # Annotation 추가
                coco_data['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': class_id,  # YOLO class ID 그대로 사용
                    'segmentation': [polygon],
                    'bbox': [x_min, y_min, bbox_w, bbox_h],
                    'area': bbox_w * bbox_h,
                    'iscrowd': 0
                })
                ann_id += 1
        
        img_id += 1
    
    print(f"✅ 변환 완료: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    
    # 저장
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    return coco_data

def process_all_splits(data_root, is_yolo=False):
    """모든 split 처리 (train/valid/test)"""
    data_root = Path(data_root)
    
    for split in ['train', 'valid', 'test']:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"⚠️ {split} 폴더 없음")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {split}")
        print('='*50)
        
        # YOLO인 경우 먼저 변환
        if is_yolo:
            coco_file = split_dir / '_annotations.coco.json'
            if not coco_file.exists():
                print(f"YOLO → COCO 변환 중...")
                yaml_file = data_root / 'data.yaml'
                convert_yolo_to_coco(split_dir, coco_file, yaml_file)
        else:
            coco_file = split_dir / '_annotations.coco.json'
        
        if coco_file.exists():
            # Blade만 추출
            blade_result = extract_blade_only(
                str(coco_file),
                str(split_dir / 'blade_only.json')
            )
            
            # 손상만 추출
            damage_result = extract_damage_only(
                str(coco_file),
                str(split_dir / 'damage_only.json')
            )
            
            if blade_result is None and damage_result is None:
                print(f"⚠️ {split}에 유효한 데이터 없음")
        else:
            print(f"❌ {coco_file} 파일 없음")

def check_categories(json_file):
    """JSON 파일의 카테고리 확인"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n파일: {Path(json_file).name}")
    print("카테고리:")
    for cat in data['categories']:
        print(f"  ID {cat['id']}: {cat['name']}")
    print(f"Annotations: {len(data.get('annotations', []))}개")