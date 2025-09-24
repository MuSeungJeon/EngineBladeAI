# review_all_damage_labels.py
import sys
import json
import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # 백엔드 설정
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from collections import defaultdict
import copy

class UnifiedDamageReviewer:
    def __init__(self, data_sources, output_root):
        """초기화"""
        self.data_sources = data_sources
        self.output_root = Path(output_root)
        self.damage_types = ['Crack', 'Nick', 'Tear']
        self.all_data = []
        self.load_all_damage_data()
        self.review_status = {}
        
        # matplotlib 설정
        plt.ion()  # Interactive mode ON
        self.current_figure = None  # 현재 figure 저장
        
    def load_all_damage_data(self):
        """모든 소스에서 damage_only.json 파일만 로드"""
        print("손상 데이터 로드 중...")
        print("="*60)
        
        total_loaded = 0
        
        for source_name, source_root in self.data_sources:
            source_root = Path(source_root)
            source_count = 0
            
            for split in ['train', 'valid', 'test']:
                damage_file = source_root / split / 'damage_only.json'
                
                if not damage_file.exists():
                    print(f"  {source_name}/{split}: damage_only.json 없음")
                    continue
                
                with open(damage_file, 'r') as f:
                    data = json.load(f)
                
                images_dict = {img['id']: img for img in data['images']}
                
                for ann_idx, ann in enumerate(data['annotations']):
                    if ann['image_id'] not in images_dict:
                        continue
                    
                    img_info = images_dict[ann['image_id']]
                    data_dir = source_root / split
                    
                    self.all_data.append({
                        'source': source_name,
                        'original_split': split,
                        'data_dir': data_dir,
                        'annotation': ann,
                        'image': img_info,
                        'global_idx': len(self.all_data),
                        'categories': data['categories'],
                        'damage_file': damage_file
                    })
                    source_count += 1
                
                print(f"  {source_name}/{split}: {len(data['annotations'])}개 손상 annotations")
            
            total_loaded += source_count
            print(f"{source_name} 총: {source_count}개")
            print("-"*40)
        
        print(f"\n✅ 총 {len(self.all_data)}개 손상 annotation 로드 완료")
        
    def review_annotation(self, idx):
        """손상 annotation 표시 - 창을 열어둔 채로"""
        if idx >= len(self.all_data) or idx < 0:
            print(f"잘못된 인덱스: {idx}")
            return None
        
        item = self.all_data[idx]
        ann = item['annotation']
        img_info = item['image']
        data_dir = item['data_dir']
        
        print(f"\n[{item['source']}/{item['original_split']}] ", end="")
        
        # 기존 라벨 정보 표시
        print("\n" + "-"*40)
        print("📌 기존 라벨 정보:")
        
        if 'category_id' in ann:
            cat_name = "Unknown"
            for cat in item['categories']:
                if cat['id'] == ann['category_id']:
                    cat_name = cat['name']
                    break
            print(f"  원본 카테고리: {cat_name}")
        
        if 'multilabel' in ann:
            labels = ann['multilabel']
            if len(labels) >= 3:
                selected = [self.damage_types[i] for i in range(min(3, len(labels))) if labels[i]]
                if selected:
                    print(f"  멀티라벨: {', '.join(selected)}")
                else:
                    print(f"  멀티라벨: 손상 없음")
        
        if 'reviewed' in ann and ann['reviewed']:
            print(f"  상태: ✓ 검토됨")
        else:
            print(f"  상태: 미검토")
        
        if idx in self.review_status:
            status = self.review_status[idx]
            if status.get('deleted'):
                print("  ⚠️ 삭제 표시됨")
            elif 'multilabel' in status:
                selected = [self.damage_types[i] for i, v in enumerate(status['multilabel']) if v]
                if selected:
                    print(f"  ✓ 현재 검토: {', '.join(selected)}")
                else:
                    print("  ✓ 현재 검토: 손상 없음")
        
        print("-"*40)
        
        try:
            img_path = data_dir / img_info['file_name']
            if not img_path.exists():
                img_path = data_dir / 'images' / img_info['file_name']
            
            if not img_path.exists():
                print(f"❌ 이미지 없음: {img_info['file_name']}")
                return None
            
            img = Image.open(img_path)
            img_array = np.array(img)
            
            h, w = img_info['height'], img_info['width']
            if 'segmentation' in ann and ann['segmentation']:
                rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                mask = maskUtils.decode(rle)
                if len(mask.shape) == 3:
                    mask = mask.sum(axis=2)
            else:
                mask = np.zeros((h, w))
            
            # 기존 figure가 있으면 재사용, 없으면 새로 생성
            if self.current_figure is None or not plt.fignum_exists(self.current_figure.number):
                self.current_figure = plt.figure(figsize=(15, 5))
            else:
                self.current_figure.clear()
            
            axes = self.current_figure.subplots(1, 3)
            
            self.current_figure.suptitle(
                f'[{idx}/{len(self.all_data)-1}] {item["source"]}/{item["original_split"]}',
                fontsize=14
            )
            
            axes[0].imshow(img_array)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Damage Mask')
            axes[1].axis('off')
            
            overlay = img_array.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.draw()
            plt.pause(0.001)
            
            if self.current_figure.canvas.manager:
                self.current_figure.canvas.manager.window.lift()
                
        except Exception as e:
            print(f"❌ 오류: {e}")
            return None
            
        return item
    
    def interactive_review(self, start_idx=0):
        """손상 데이터 검토 - 창 유지 버전"""
        idx = start_idx
        
        print("\n" + "="*60)
        print("통합 손상 라벨 검토 시스템")
        print(f"총 {len(self.all_data)}개 손상 annotations")
        print("손상 유형: Crack, Nick, Tear")
        print("="*60)
        print("\n명령어:")
        print("  1-3: 손상 선택 (1=Crack, 2=Nick, 3=Tear)")
        print("  여러 개: 1 2 또는 1,2")
        print("  0: 손상 없음")
        print("  Enter: 기존 라벨 유지")
        print("  d: 삭제")
        print("  s/n: 다음")
        print("  b: 이전")
        print("  g[번호]: 이동 (예: g100)")
        print("  status: 진행 상황")
        print("  save: 저장")
        print("  q: 종료")
        print("="*60)
        print("\n💡 팁: 이미지 창은 자동으로 업데이트됩니다. 닫지 마세요!")
        print("="*60)
        
        while True:
            print(f"\n{'='*60}")
            print(f"현재 위치: [{idx}/{len(self.all_data)-1}]")
            print('='*60)
            
            item = self.review_annotation(idx)
            
            if item is None:
                print("항목을 표시할 수 없습니다.")
                user_input = input(">>> 다음(s) 또는 이전(b): ").strip().lower()
            else:
                existing_label = None
                ann = item['annotation']
                
                if 'multilabel' in ann:
                    existing_label = ann['multilabel']
                elif 'category_id' in ann:
                    existing_label = [0, 0, 0]
                    cat_name = None
                    for cat in item['categories']:
                        if cat['id'] == ann['category_id']:
                            cat_name = cat['name']
                            break
                    if cat_name in self.damage_types:
                        idx_damage = self.damage_types.index(cat_name)
                        existing_label[idx_damage] = 1
                
                print("\n" + "-"*40)
                print("선택 옵션:")
                print("  1=Crack  2=Nick  3=Tear")
                print("  0=손상없음  d=삭제  s=다음  b=이전")
                
                if existing_label:
                    existing_names = [self.damage_types[i] for i, v in enumerate(existing_label[:3]) if v]
                    if existing_names:
                        print(f"  Enter=기존 유지 ({', '.join(existing_names)})")
                    else:
                        print(f"  Enter=기존 유지 (손상 없음)")
                
                print("-"*40)
                
                print(">>> 입력하세요: ", end="", flush=True)
                user_input = input().strip().lower()
                
                if user_input == "" and existing_label is not None:
                    self.review_status[idx] = {'multilabel': existing_label[:3]}
                    selected = [self.damage_types[i] for i, v in enumerate(existing_label[:3]) if v]
                    if selected:
                        print(f"    [✓ 기존 유지: {', '.join(selected)}]")
                    else:
                        print(f"    [✓ 기존 유지: 손상 없음]")
                    if idx < len(self.all_data) - 1:
                        idx += 1
                    continue
            
            print(f"    [입력됨: {user_input if user_input else '(Enter)'}]")
            
            if user_input == 'q':
                if self.current_figure:
                    plt.close(self.current_figure)
                    
                confirm = input(">>> 정말 종료? 저장하고 종료(y) / 그냥 종료(n) / 취소(c): ")
                if confirm.lower() == 'y':
                    self.save_and_redistribute()
                    break
                elif confirm.lower() == 'n':
                    break
                else:
                    print("    [계속 진행]")
                    continue
                    
            elif user_input == 'save':
                print(">>> 저장 중...")
                self.save_and_redistribute()
                print("    [✓ 저장 완료]")
                
            elif user_input == 'status':
                reviewed = len(self.review_status)
                deleted = sum(1 for s in self.review_status.values() if s.get('deleted'))
                print(f"\n{'='*40}")
                print(f"진행 상황:")
                print(f"  전체: {len(self.all_data)}개")
                print(f"  검토: {reviewed}개 ({reviewed/len(self.all_data)*100:.1f}%)")
                print(f"  삭제: {deleted}개")
                print(f"  남음: {len(self.all_data) - reviewed}개")
                print('='*40)
                
            elif user_input in ['s', 'n']:
                if idx < len(self.all_data) - 1:
                    idx += 1
                    print(f"    [→ 다음: {idx}번으로 이동]")
                else:
                    print("    [마지막 항목입니다]")
                    
            elif user_input == 'b':
                if idx > 0:
                    idx -= 1
                    print(f"    [← 이전: {idx}번으로 이동]")
                else:
                    print("    [첫 번째 항목입니다]")
                    
            elif user_input.startswith('g'):
                try:
                    target = int(user_input[1:])
                    if 0 <= target < len(self.all_data):
                        idx = target
                        print(f"    [점프: {target}번으로 이동]")
                    else:
                        print(f"    [❌ 범위 초과: 0-{len(self.all_data)-1}만 가능]")
                except:
                    print("    [❌ 잘못된 형식. 예: g100]")
                    
            elif user_input == 'd':
                self.review_status[idx] = {'deleted': True}
                print("    [🗑️ 삭제 표시됨]")
                if idx < len(self.all_data) - 1:
                    idx += 1
                    
            elif user_input == '0':
                self.review_status[idx] = {'multilabel': [0, 0, 0]}
                print("    [✓ 손상 없음으로 표시]")
                if idx < len(self.all_data) - 1:
                    idx += 1
                    
            else:
                multilabel = [0, 0, 0]
                valid_input = False
                
                for num in user_input.replace(',', ' ').split():
                    try:
                        n = int(num) - 1
                        if 0 <= n < 3:
                            multilabel[n] = 1
                            valid_input = True
                    except:
                        pass
                
                if valid_input and any(multilabel):
                    self.review_status[idx] = {'multilabel': multilabel}
                    selected = [self.damage_types[i] for i, v in enumerate(multilabel) if v]
                    print(f"    [✓ 선택: {', '.join(selected)}]")
                    if idx < len(self.all_data) - 1:
                        idx += 1
                else:
                    print("    [❌ 잘못된 입력. 1-3 숫자를 입력하세요]")
    
    def save_and_redistribute(self):
        """검토 완료 후 6:2:2로 재분배하여 저장"""
        print("\n데이터 재분배 중...")
        
        valid_data = []
        for idx, item in enumerate(self.all_data):
            if idx in self.review_status:
                status = self.review_status[idx]
                if status.get('deleted'):
                    continue
                if 'multilabel' in status:
                    item['annotation']['multilabel'] = status['multilabel']
                    item['annotation']['reviewed'] = True
            valid_data.append(item)
        
        random.shuffle(valid_data)
        
        total = len(valid_data)
        train_end = int(total * 0.6)
        valid_end = int(total * 0.8)
        
        splits = {
            'train': valid_data[:train_end],
            'valid': valid_data[train_end:valid_end],
            'test': valid_data[valid_end:]
        }
        
        print(f"\n재분배 결과:")
        print(f"  Train: {len(splits['train'])}개")
        print(f"  Valid: {len(splits['valid'])}개")
        print(f"  Test: {len(splits['test'])}개")
        
        for split_name, split_data in splits.items():
            self.save_split(split_name, split_data)
        
        print(f"\n✅ 모든 데이터가 {self.output_root}에 저장됨")
    
    def save_split(self, split_name, data_items):
        """각 split을 damage_multilabel.json으로 저장"""
        output_dir = self.output_root / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        coco_format = {
            'info': {'description': f'Multilabel damage detection - {split_name}'},
            'licenses': [{'id': 1, 'name': 'CC BY 4.0'}],
            'categories': [
                {'id': i, 'name': name, 'supercategory': 'damage'} 
                for i, name in enumerate(self.damage_types)
            ],
            'images': [],
            'annotations': []
        }
        
        img_id_map = {}
        new_img_id = 0
        new_ann_id = 0
        
        for item in data_items:
            orig_img = item['image']
            orig_ann = item['annotation']
            
            orig_img_key = f"{item['source']}_{orig_img['id']}_{orig_img['file_name']}"
            
            if orig_img_key not in img_id_map:
                src_img = item['data_dir'] / orig_img['file_name']
                if not src_img.exists():
                    src_img = item['data_dir'] / 'images' / orig_img['file_name']
                
                if src_img.exists():
                    dst_img = output_dir / orig_img['file_name']
                    
                    if dst_img.exists():
                        base = dst_img.stem
                        ext = dst_img.suffix
                        counter = 1
                        while dst_img.exists():
                            dst_img = output_dir / f"{base}_{counter}{ext}"
                            counter += 1
                    
                    shutil.copy2(src_img, dst_img)
                    final_filename = dst_img.name
                else:
                    print(f"  ⚠️ 이미지 없음: {orig_img['file_name']}")
                    continue
                
                img_id_map[orig_img_key] = new_img_id
                
                coco_format['images'].append({
                    'id': new_img_id,
                    'file_name': final_filename,
                    'height': orig_img['height'],
                    'width': orig_img['width']
                })
                new_img_id += 1
            
            new_ann = orig_ann.copy()
            new_ann['id'] = new_ann_id
            new_ann['image_id'] = img_id_map[orig_img_key]
            
            if 'multilabel' not in new_ann:
                multilabel = [0, 0, 0]
                
                cat_name = None
                for cat in item['categories']:
                    if cat['id'] == new_ann.get('category_id', -1):
                        cat_name = cat['name']
                        break
                
                if cat_name in self.damage_types:
                    idx = self.damage_types.index(cat_name)
                    multilabel[idx] = 1
                
                new_ann['multilabel'] = multilabel
            
            if 'category_id' in new_ann:
                del new_ann['category_id']
            
            coco_format['annotations'].append(new_ann)
            new_ann_id += 1
        
        json_path = output_dir / 'damage_multilabel.json'
        with open(json_path, 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        print(f"  {split_name}: {len(coco_format['images'])}개 이미지, {len(coco_format['annotations'])}개 annotation")
        
        multi_count = 0
        for ann in coco_format['annotations']:
            if 'multilabel' in ann and sum(ann['multilabel']) > 1:
                multi_count += 1
        
        if multi_count > 0:
            print(f"    멀티라벨 (복합 손상): {multi_count}개")

# ===== MAIN 실행 부분 =====
if __name__ == "__main__":
    # 데이터 소스 정의
    data_sources = [
        ('YOLO2', r'C:\EngineBladeAI\EngineInspectionAI_MS\data\final_ver_data_rev2.v3i.yolov8'),
        ('COCO_SEG', r'C:\EngineBladeAI\EngineInspectionAI_MS\data\coco_segmentation_for_multilabel')
    ]
    
    # 출력 경로
    output_root = r'C:\EngineBladeAI\EngineInspectionAI_MS\data\multilabeled_data'
    
    # 리뷰어 생성
    reviewer = UnifiedDamageReviewer(data_sources, output_root)
    
    # 시작 인덱스
    start = input(f"시작 인덱스 (Enter=0, 최대={len(reviewer.all_data)-1}): ").strip()
    start_idx = int(start) if start else 0
    
    # 검토 시작
    reviewer.interactive_review(start_idx)