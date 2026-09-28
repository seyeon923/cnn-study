import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):
    def __init__(
        self, input_channels: int = 3, num_classes: int = 1000, width_multiplier: float = 1.0
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                self._get_channel(32, width_multiplier),
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self._get_channel(32, width_multiplier)),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(
                self._get_channel(32, width_multiplier),
                self._get_channel(64, width_multiplier),
                stride=1,
            ),
            DepthwiseSeparableConv(
                self._get_channel(64, width_multiplier),
                self._get_channel(128, width_multiplier),
                stride=2,
            ),
            DepthwiseSeparableConv(
                self._get_channel(128, width_multiplier),
                self._get_channel(128, width_multiplier),
                stride=1,
            ),
            DepthwiseSeparableConv(
                self._get_channel(128, width_multiplier),
                self._get_channel(256, width_multiplier),
                stride=2,
            ),
            DepthwiseSeparableConv(
                self._get_channel(256, width_multiplier),
                self._get_channel(256, width_multiplier),
                stride=1,
            ),
            DepthwiseSeparableConv(
                self._get_channel(256, width_multiplier),
                self._get_channel(512, width_multiplier),
                stride=2,
            ),
            *[
                DepthwiseSeparableConv(
                    self._get_channel(512, width_multiplier),
                    self._get_channel(512, width_multiplier),
                    stride=1,
                )
                for _ in range(5)
            ],
            DepthwiseSeparableConv(
                self._get_channel(512, width_multiplier),
                self._get_channel(1024, width_multiplier),
                stride=2,
            ),
            DepthwiseSeparableConv(
                self._get_channel(1024, width_multiplier),
                self._get_channel(1024, width_multiplier),
                stride=1,
            ),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self._get_channel(1024, width_multiplier), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _get_channel(self, base_channel: int, width_multiplier: float) -> int:
        return max(round(base_channel * width_multiplier), 1)


if __name__ == "__main__":
    from thop import profile

    model = MobileNet(input_channels=3, num_classes=1000, width_multiplier=1.0)

    x = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(x,))
    print(f"MACs: {macs:,}")
    print(f"Params: {params:,}")
