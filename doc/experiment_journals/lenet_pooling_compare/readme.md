# LeNet-5 Activation Comparision

Compare 2 different poolings(avg vs max) for LeNet-5 with CIFAR-10 dataset.

## Experiment Environment

- OS: Ubuntu 24.04.4 LTS on Windows 10 x86_64
- Kernel: 5.15.153.1-microsoft-standard-WSL2
- CPU: 13th Gen Intel i9-13900H (20) @ 2.995GHz
- Memory: 5003MiB / 23898MiB
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- Git Commit: b864af33d3a8a8893eeb5e0a435992e72fa24d1d

## Test Results

### Test 1(w/o Augmentation)

|Pooling|Accuracy Mean(%)|Accuracy Std(%)|
|---|---|---|
|Average|64.24|0.90|
|Max|**65.30**|0.73|

> 10 repeated tests are done for each pooling

- Max pooling shows slightly better performance on this setting(about 1%p better than average pooling)
- After 10 epochs, both shows evident overfitting

#### Hyper Parameters
- Optimizer: AdamW
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 30
- Augmentation: No

#### Commands

- **Average Pooling**

  ```bash
  uv run python scripts/train.py data=cifar10 data.augmentation=false model=lenet5 model.input_channels=3 model.pooling=avg experiment_name=lenet_avg_pool
  ```

- **Max Pooling**

  ```bash
  uv run python scripts/train.py data=cifar10 data.augmentation=false model=lenet5 model.input_channels=3 model.pooling=max experiment_name=lenet_max_pool
  ```


### Test 2(w/ Augmentation)

- Since overfitting was observed in Test 1, an additional experiment was conducted with data augmentation (random horizontal/vertical flip and grayscale transformation).
- To reduce the impact of random initialization, each pooling configuration was evaluated over 10 independent runs.


|Pooling|Accuracy Mean(%)|Accuracy Std(%)|
|---|---|---|
|Average|**66.18**|0.81|
|Max|65.77|0.60|

#### Hyper Parameters
- Optimizer: AdamW
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 100
- Augmetation: Yes

#### Commands

- **Average Pooling**

  ```bash
  uv run python scripts/train.py data=cifar10 data.augmentation=true model=lenet5 model.input_channels=3 model.pooling=avg experiment_name=lenet_avg_pool
  ```

- **Max Pooling**

  ```bash
  uv run python scripts/train.py data=cifar10 data.augmentation=true model=lenet5 model.input_channels=3 model.pooling=max experiment_name=lenet_max_pool
  ```

## Conclusion


- Without augmentation, Max Pooling achieved slightly higher accuracy than Average Pooling (~1.1%p).
- After applying data augmentation, the performance gap between the two pooling methods became smaller, and Average Pooling showed a slight advantage (~0.4%p).
- The observed differences are relatively small compared to the run-to-run variation, suggesting that pooling type is not a dominant factor for LeNet-5 on CIFAR-10.
- Data augmentation had a larger impact on performance than the choice of pooling method.
- For this experimental setting, both pooling methods are viable, with Average Pooling showing a small generalization benefit when augmentation is used.
- These results should not be generalized to modern CNN architectures, as the experiment was conducted using the relatively small LeNet-5 architecture on CIFAR-10.