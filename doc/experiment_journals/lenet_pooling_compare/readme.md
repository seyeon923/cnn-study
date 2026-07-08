# LeNet-5 Activation Comparision

Compare 2 different poolings(avg vs max) for LeNet-5 with CIFAR-10 dataset.

## Experiment Environment

- OS: Ubuntu 24.04.4 LTS on Windows 10 x86_64
- Kernel: 5.15.153.1-microsoft-standard-WSL2
- CPU: 13th Gen Intel i9-13900H (20) @ 2.995GHz
- Memory: 5003MiB / 23898MiB
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- Git Commit: 951c623ffa36086f68e3455d49870c08a92a92df

## Test Results

|Pooling|Accuracy Mean(%)|Accuracy Std(%)|
|---|---|---|
|Average|64.24|0.90|
|Max|65.30|0.73|

> 10 repeated tests are done for each pooling

- Max pooling shows slightly better performance on this setting(about 1%p better than average pooling)
- After 10 epochs, both shows evident overfitting

### Hyper Parameters
- Optimizer: AdamW
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 30

### Commands

- **Average Pooling**

  ```bash
  uv run python scripts/train.py data=cifar10 data.augmentation=false model=lenet5 model.input_channels=3 model.pooling=avg experiment_name=lenet_avg_pool
  ```

- **Max Pooling**

  ```bash
  uv run python scripts/train.py data=cifar10 data.augmentation=false model=lenet5 model.input_channels=3 model.pooling=max experiment_name=lenet_max_pool
  ```

> 추가 실험: augmetation 적용해서 epoch 크게 수행
> - 100 epoch 씩 해봤을 땐, 큰 overfitting 은 보이진 않았고, avg pooling 은 66.6%, max pooling 은 65.70% 정확도 보임
> - max pooling은 50-60 epoch 부근 부터 overfitting 이 보이는 거 같기도 한데, 추가적인 실험 필요해 보임.(avg vs max pooling 1%p 차이라 반복 테스트 필요해보임)