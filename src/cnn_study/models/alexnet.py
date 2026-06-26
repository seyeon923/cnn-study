import torch
from torch import nn


class AlexNet(nn.Module):
    """Modernized AlexNet with flexible input size via adaptive pooling."""

    def __init__(
        self,
        input_channels: int = 3,
        output_classes: int = 1000,
        use_bn: bool = False,
        use_final_dropout: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_classes = output_classes
        self.use_bn = use_bn
        self.use_final_dropout = use_final_dropout

        layers = []

        # Conv block 1
        layers.append(nn.Conv2d(input_channels, 96, kernel_size=11, stride=4))
        if use_bn:
            layers.append(nn.BatchNorm2d(96))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # Conv block 2
        layers.append(nn.Conv2d(96, 256, kernel_size=5, padding=2))
        if use_bn:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # Conv block 3
        layers.append(nn.Conv2d(256, 384, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(384))
        layers.append(nn.ReLU(inplace=True))

        # Conv block 4
        layers.append(nn.Conv2d(384, 384, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(384))
        layers.append(nn.ReLU(inplace=True))

        # Conv block 5
        layers.append(nn.Conv2d(384, 256, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        classfier_layers = [
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        ]
        if use_final_dropout:
            classfier_layers.append(nn.Dropout(p=0.5))
        classfier_layers.append(nn.Linear(4096, output_classes))

        self.classifer = nn.Sequential(*classfier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == "__main__":
    from thop import profile

    model = AlexNet(3, 1000, use_bn=True)

    x = torch.randn(1, 3, 227, 227)
    y = model(x)

    macs, params = profile(model, inputs=(x,))

    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"MACs: {macs:,}")
    print(f"Params: {params:,}")
