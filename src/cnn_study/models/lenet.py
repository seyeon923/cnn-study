import torch
from torch import nn

from .classifier import Classifier, make_activation


class LeNet5(nn.Module):
    """Modernized LeNet with flexible input size.

    Notes:
        - Works with variable input spatial sizes as long as they are large enough.
        - Uses AdaptiveAvgPool2d((5, 5)) before the classifier so the classifier
          input dimension remains fixed.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_classes: int = 10,
        activation: str = "relu",
        pooling: str = "avg",
        use_bn: bool = False,
        classifier_type: str = "conv_dense",
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_classes = output_classes
        self.activation_name = activation
        self.pooling_name = pooling
        self.use_bn = use_bn

        layers = []

        # Conv block 1
        layers.append(nn.Conv2d(input_channels, 6, kernel_size=5, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(6))
        layers.append(make_activation(activation))
        layers.append(self._make_pool())

        # Conv block 2
        layers.append(nn.Conv2d(6, 16, kernel_size=5, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(16))
        layers.append(make_activation(activation))
        layers.append(self._make_pool())

        self.features = nn.Sequential(*layers)

        self.classifier = Classifier(
            16,
            output_classes,
            hidden_features=[120, 84],
            num_hidden_layers=2,
            expected_feature_size=4,  # 4 x 4 for 28 x 28 input
            classifier_type=classifier_type,
            activation=activation,
        )

    def _make_pool(self) -> nn.Module:
        if self.pooling_name == "avg":
            return nn.AvgPool2d(kernel_size=2)
        elif self.pooling_name == "max":
            return nn.MaxPool2d(kernel_size=2)
        else:
            raise ValueError(f"Not supported pooling: {self.pooling_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    from thop import profile

    model = LeNet5(
        input_channels=3,
        output_classes=10,
        activation="relu",
        pooling="avg",
        use_bn=True,
    )

    for size in [28, 32, 36, 48, 64]:
        x = torch.randn(4, 3, size, size)
        y = model(x)

        macs, params = profile(model, inputs=(x,))

        print(f"Input: {x.shape}")
        print(f"Output: {y.shape}")
        print(f"MACs: {macs:,}")
        print(f"Params: {params:,}")
