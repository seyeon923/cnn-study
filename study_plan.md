# Study Plan

목표:

- [ ] 모델을 "사용"하는 수준에서 "구조를 이해"하는 수준으로 발전
- [ ] Task별 대표 모델을 체계적으로 정리
- [ ] 논문 + 구조 이해 + 실습을 반복하여 실무 감각 향상
- [ ] Accuracy / FLOPs / Params / Latency 관점에서 모델 비교 가능해지기

---

# Phase 1. CNN Backbone 이해 (최우선)

목표:

- CNN 구조 진화 과정 이해
- Classification Backbone 비교 가능
- 모델별 설계 의도 이해

## 1.1 LeNet

### 학습

- [x] 논문 초록 및 Introduction 확인
- [x] 모델 구조 정리
- [x] Modernized LeNet 구현
- [x] CIFAR10 학습 실험
- [x] Activation 별 비교 (ReLU / Tanh / Sigmoid)
- [x] Pooling 별 비교 (AvgPool / MaxPool)
- [ ] BatchNorm 유무 비교

### 확인할 것

- [x] Local Receptive Field
- [x] Weight Sharing
- [x] CNN 등장 배경 이해

---

## 1.2 AlexNet

### 학습

- [x] 논문 핵심 포인트 정리
- [x] Modernized AlexNet 구현
- [ ] CIFAR10 또는 TinyImageNet 학습
- [ ] LeNet 대비 성능 비교
- [ ] Final layer dropout 추가 성능 비교

### 확인할 것

- [x] ReLU
- [x] Dropout
- [x] Data Augmentation
- [x] Overlapping Pooling
- [x] LRN 개념 이해
- [x] GPU 병렬화 배경 이해

---

## 1.3 VGG

### 학습

- [x] 논문 읽기
- [x] VGG16 구현
- [x] VGG19 구현
- [ ] AlexNet 비교

### 확인할 것

- [x] 작은 Kernel (3×3) 반복 사용
- [x] Deep CNN 효과

---

## 1.4 ResNet ⭐

### 학습

- [ ] 논문 읽기
- [ ] Residual Block 구현
- [ ] ResNet18 구현
- [ ] ResNet34 구현
- [ ] CIFAR10 학습

### 확인할 것

- [ ] Skip Connection
- [ ] Gradient Vanishing 해결
- [ ] Deep Network 학습 가능 이유

---

## 1.5 MobileNet

### 학습

- [ ] MobileNet V1 구조 분석
- [ ] MobileNet V2 구조 분석
- [ ] MBConv 이해
- [ ] CIFAR10 학습

### 확인할 것

- [ ] Depthwise Separable Convolution
- [ ] Inverted Residual
- [ ] Linear Bottleneck

---

## 1.6 EfficientNet

### 학습

- [ ] 논문 읽기
- [ ] EfficientNet-B0 구조 분석
- [ ] CIFAR10 학습
- [ ] MobileNet 비교

### 확인할 것

- [ ] Compound Scaling
- [ ] MBConv
- [ ] SE Block

---

## 1.7 ConvNeXt (선택)

### 학습

- [ ] 논문 읽기
- [ ] 구조 분석

### 확인할 것

- [ ] CNN의 Transformer화
- [ ] Modern CNN Design

---

## Backbone 실험 과제

### 동일 조건 비교

Dataset:

- [ ] CIFAR10
- [ ] ImageNet

Models:

- [ ] LeNet
- [ ] AlexNet
- [ ] VGG
- [ ] ResNet18
- [ ] MobileNetV2
- [ ] EfficientNet-B0

기록:

- [ ] Accuracy
- [ ] FLOPs
- [ ] Params
- [ ] CPU Latency
- [ ] GPU Latency

---

# Phase 2. Detection & Segmentation

## 2.1 Object Detection

### Faster R-CNN

- [ ] 논문 읽기
- [ ] 구조 이해
- [ ] RPN 이해

### YOLO

- [ ] YOLOv5
- [ ] YOLOv8
- [ ] 실습

### DETR (선택)

- [ ] 논문 읽기
- [ ] 구조 이해

---

## 2.2 Segmentation

### U-Net ⭐

- [ ] 논문 읽기
- [ ] 구현
- [ ] 실습

확인할 것:

- [ ] Encoder-Decoder
- [ ] Skip Connection

### DeepLabV3+

- [ ] 논문 읽기
- [ ] 구조 이해

확인할 것:

- [ ] ASPP

### Mask R-CNN (선택)

- [ ] 구조 이해

---

## Segmentation 실습

- [ ] Defect Segmentation Dataset 확보
- [ ] U-Net 실습
- [ ] Detection vs Segmentation 비교

---

# Phase 3. Landmark / Keypoint Detection

## PFLD

- [ ] 논문 재분석
- [ ] 구조 분석
- [ ] 코드 재검토

## HRNet

- [ ] 논문 읽기
- [ ] 구조 이해

## SimpleBaseline

- [ ] 논문 읽기
- [ ] 구조 이해

### 학습 포인트

- [ ] Heatmap 기반 방법
- [ ] Direct Regression 기반 방법
- [ ] 장단점 비교

---

# Phase 4. Anomaly Detection ⭐

## Reconstruction 기반

### AutoEncoder

- [ ] 구현
- [ ] MVTec 실습

### VAE

- [ ] 구현
- [ ] 실습

---

## Feature 기반

### PaDiM

- [ ] 논문 읽기
- [ ] 구조 분석
- [ ] 실습

### PatchCore ⭐

- [ ] 논문 읽기
- [ ] 구현 또는 코드 분석
- [ ] 실습

### EfficientAD

- [ ] 논문 재정리
- [ ] Teacher-Student 구조 분석
- [ ] Feature Distillation 분석

---

## 실습

- [ ] MVTec Dataset
- [ ] AUROC 비교
- [ ] Inference 속도 비교

---

# Phase 5. Transformer 기반 Vision Model

## ViT

- [ ] 논문 읽기
- [ ] 구조 이해
- [ ] 실습

확인할 것:

- [ ] Patch Embedding
- [ ] Self Attention

---

## Swin Transformer

- [ ] 논문 읽기
- [ ] 구조 이해

확인할 것:

- [ ] Window Attention
- [ ] Hierarchical Structure

---

## SAM (선택)

- [ ] 논문 읽기
- [ ] 구조 이해

---

# Phase 6. Optimization & Deployment

## Quantization

- [ ] PTQ
- [ ] QAT

---

## Pruning

- [ ] Structured Pruning
- [ ] Unstructured Pruning

---

## Knowledge Distillation

- [ ] Teacher-Student
- [ ] Distillation Loss

---

## Deployment

- [ ] ONNX Export
- [ ] TensorRT
- [ ] CPU Inference Benchmark

---

# 반복 루틴 (모델 하나당)

- [ ] 논문 Skim (30~60분)
- [ ] 구조 다이어그램 이해
- [ ] PyTorch 직접 구현 또는 코드 추적
- [ ] Dataset 학습
- [ ] Accuracy 측정
- [ ] FLOPs 측정
- [ ] Params 측정
- [ ] Latency 측정
- [ ] README 정리

---

# 8주 추천 진행 계획

## Week 1 ~ 2

- [ ] ResNet 완전 이해
- [ ] ResNet18 구현
- [ ] CIFAR10 실험

## Week 3

- [ ] MobileNet 구조 분석
- [ ] EfficientNet 구조 분석
- [ ] MobileNet ↔ EfficientNet 비교

## Week 4

- [ ] U-Net 실습

## Week 5

- [ ] YOLO 실습

## Week 6

- [ ] PatchCore 실습

## Week 7

- [ ] ViT 이해 및 실습

## Week 8

- [ ] Quantization
- [ ] ONNX Export
- [ ] TensorRT

---

# 최종 목표

## 모델 구조 이해

- [ ] 왜 이런 구조가 나왔는지 설명 가능
- [ ] FLOPs/Params 변화 예측 가능
- [ ] 새로운 논문 구조 빠르게 이해 가능

## Task별 모델 Mapping

### Classification

- [ ] ResNet
- [ ] MobileNet
- [ ] EfficientNet

### Detection

- [ ] Faster R-CNN
- [ ] YOLO

### Segmentation

- [ ] U-Net
- [ ] DeepLabV3+

### Landmark

- [ ] PFLD
- [ ] HRNet

### Anomaly Detection

- [ ] PatchCore
- [ ] PaDiM
- [ ] EfficientAD

### Transformer

- [ ] ViT
- [ ] Swin Transformer

## 실험 감각 확보

- [ ] Learning Rate 영향 이해
- [ ] Optimizer 영향 이해
- [ ] Dataset 영향 이해
- [ ] Augmentation 영향 이해
- [ ] Noise 영향 이해
- [ ] Accuracy / FLOPs / Latency Trade-off 이해