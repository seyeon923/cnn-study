# CNN Study

Study and implementation of CNN architectures (LeNet, AlexNet, VGG, ResNet, MobileNet, EfficientNet) with PyTorch, along with structured experiments and performance comparisons.

## CNN Architectures

- [Lenet-5](./doc/lenet.md)
  - 초창기 성공적인 CNN 구조, 이미지(MNIST 필기 인식)에서 좋은 성능을 보여줌.
  - Convolution 연산으로 이미지에서 공간적인 정보를 유지(local receptive fields)
  - Convolution 연산으로 여러 위치(픽셀)에서 공유하는 weights 사용(parameter 수 감소)

- [AlexNet](./doc/alexnet.md)
  - ILSVRC-2012 imagenet classfication 우승
  - Deep CNN 구조가 large-scale dataset(예: imagenet)에서 좋은 성능을 낼 수 있음을 보여줌.

- [VGG](./doc/vgg.md)
  - ILSVRC-2014 localisation/classification 에서 각각 우승/준우승
  - 3x3 conv(stride=1, padding=1), 2x2 max pooling 으로 단일화된 단순한 구조만 반복하여 사용
  - layer가 깊어질수록 성능이 좋아지는 것을 보여줌.

- [ResNet](./doc/resnet.md)
  - 입/출력 층을 skip(identity) connection으로 연결하여 층이 매우 깊어질 때 오히려 optimization이 덜되는 degradation 현상 해결
  - 150층 이상의 매우 깊은 CNN 구조로 ILSVRC-2015, COCO-2015 우승 

- MobileNet: 모바일/임베디드 환경을 위한 효율적인 경량화 모델구조 제안
  - [MobileNetV1](./doc/mobilenets/mobilenet_v1.md)
    - Standard convolution 대신 depthwise separable convolution(3x3 채널별 + 1x1)을 사용
  - [MobileNetV2](./doc/mobilenets/mobilenet_v2.md)
    - Inverted residual with linear bottleneck 구조 사용
      - Bottleneck layer 간에 skip(residual) connection 연결
      - Bottleneck layer 에는 non-linear activation 제거
      - 3x3 convolution에는 depthwise convolution 사용.
      - Channel 수 변화에는(expansion/bottlenec) 1x1 conv(pointwise conv) 사용