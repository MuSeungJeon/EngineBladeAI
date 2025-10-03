main : 기본
branch
 - 1001: ConvNext-FPN , SegFormer, Mask2Former 학습 돌아감. End-to-End 학습 Head-A 학습 잘 됨. Head-B 학습 안 됨
 - 1002: ConvNext-FPN , SegFormer, Mask2Former 학습 돌아감. 학습 자체를 two stage 로 진행. Head-A 학습 잘 되고 Head-B 학습 안 됨.
 - 1003: ConvNext-FPN , SegFormer, Mask2Former, COCO 데이터로 테스트 해봄. COCODataSetTest.ipynb
Hybrid_model : ConvNext-FPN Mask2Former 만 사용, Gaussian Distance 사용, mAP 0 뜸. F1, precision, recall 은 올라감. 근데 전혀 못 맞춰서 실패
Unified_model : ConvNext-FPN Mask2Former 만 사용, Gaussian Distance 사용.
