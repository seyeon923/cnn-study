# CNN Study

Study and implementation of CNN architectures (LeNet, AlexNet, VGG, ResNet, MobileNet, EfficientNet) with PyTorch, along with structured experiments and performance comparisons.

## LeNet-5

- Paper: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- Authors: Yann LeCunn, Leon Bottou, Yoshua Bengio, and Patrick Haffner

![LeNet-5 Architecture](./doc/images/lenet_architecture.png)


- One of the earliest successful Convolutional Neural Network (CNN) architectures

- Motivation:
  - Fully connected networks are inefficient for image data (too many parameters)
  - Exploits spatial structure of images

- Key Ideas:
  - Local receptive fields
  - Weight sharing (reduces parameters)

- Significance:
  - Demonstrated strong performance on handwritten digit recognition (MNIST)
  - Foundation of modern CNN architectures



> The implementation is slightly adapted to a more modern architecture.
> 
> - S2 -> C3 feature map fully connected
> - Trainable Pooling => Fixed Pooling
> - tanh activation => ReLU activation
> - Optional batch normalization

## AlexNet

- Paper: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- Authors: Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton


- Winner of ILSVRC-2012 (ImageNet classification)
- Demonstrated that deep CNNs can achieve strong performance on large-scale datasets

- Key contributions:
  - ReLU activation (faster training than tanh/sigmoid)
  - Model parallelism across two GPUs (split network with limited cross-connections)
  - Local Response Normalization (LRN)
    - Normalizes across nearby feature channels
    - Encourages competition between neurons
  - Overlapping pooling (kernel size > stride)
  - Extensive data augmentation:
    - Random cropping / translation
    - Horizontal reflection
    - PCA-based RGB color augmentation
  - Dropout in fully connected layers to reduce overfitting

- Architectural characteristics:
  - Large convolution kernel and stride in early layers (11x11, stride 4)
  - Large fully connected layers (dominant parameter count)


> The implementation is slightly adapted to a more modern architecture.
> 
> - Full connected CNN connection(for single GPU)
> - Remove LRN(instead use optional batch normalization)