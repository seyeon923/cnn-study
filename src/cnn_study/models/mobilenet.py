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


def make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def get_channels(base_channel: int, width_multiplier: float, divisor: int = 8) -> int:
    return make_divisible(base_channel * width_multiplier, divisor)


class MobileNet(nn.Module):
    def __init__(
        self, input_channels: int = 3, num_classes: int = 1000, width_multiplier: float = 1.0
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                get_channels(32, width_multiplier),
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(get_channels(32, width_multiplier)),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(
                get_channels(32, width_multiplier),
                get_channels(64, width_multiplier),
                stride=1,
            ),
            DepthwiseSeparableConv(
                get_channels(64, width_multiplier),
                get_channels(128, width_multiplier),
                stride=2,
            ),
            DepthwiseSeparableConv(
                get_channels(128, width_multiplier),
                get_channels(128, width_multiplier),
                stride=1,
            ),
            DepthwiseSeparableConv(
                get_channels(128, width_multiplier),
                get_channels(256, width_multiplier),
                stride=2,
            ),
            DepthwiseSeparableConv(
                get_channels(256, width_multiplier),
                get_channels(256, width_multiplier),
                stride=1,
            ),
            DepthwiseSeparableConv(
                get_channels(256, width_multiplier),
                get_channels(512, width_multiplier),
                stride=2,
            ),
            *[
                DepthwiseSeparableConv(
                    get_channels(512, width_multiplier),
                    get_channels(512, width_multiplier),
                    stride=1,
                )
                for _ in range(5)
            ],
            DepthwiseSeparableConv(
                get_channels(512, width_multiplier),
                get_channels(1024, width_multiplier),
                stride=2,
            ),
            DepthwiseSeparableConv(
                get_channels(1024, width_multiplier),
                get_channels(1024, width_multiplier),
                stride=1,
            ),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(get_channels(1024, width_multiplier), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class InvertedResidualBottleneck(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, expansion_factor: int = 6
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        self.use_expansion = expansion_factor != 1

        hidden_dim = in_channels * expansion_factor

        if self.use_expansion:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)

        self.depthwise_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.use_expansion:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.relu6(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.relu6(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.use_residual:
            return x + identity

        return x


class MobileNetV2(nn.Module):
    def __init__(
        self, input_channels: int = 3, num_classes: int = 1000, width_multiplier: float = 1.0
    ):
        super().__init__()
        self.initial_conv = nn.Conv2d(
            input_channels,
            get_channels(32, width_multiplier),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.initial_bn = nn.BatchNorm2d(get_channels(32, width_multiplier))

        self.bottlenecks = nn.Sequential(
            self._get_bottlenecks(
                in_channels=get_channels(32, width_multiplier),
                out_channels=get_channels(16, width_multiplier),
                stride=1,
                expansion_factor=1,
                repeats=1,
            ),
            self._get_bottlenecks(
                in_channels=get_channels(16, width_multiplier),
                out_channels=get_channels(24, width_multiplier),
                stride=2,
                expansion_factor=6,
                repeats=2,
            ),
            self._get_bottlenecks(
                in_channels=get_channels(24, width_multiplier),
                out_channels=get_channels(32, width_multiplier),
                stride=2,
                expansion_factor=6,
                repeats=3,
            ),
            self._get_bottlenecks(
                in_channels=get_channels(32, width_multiplier),
                out_channels=get_channels(64, width_multiplier),
                stride=2,
                expansion_factor=6,
                repeats=4,
            ),
            self._get_bottlenecks(
                in_channels=get_channels(64, width_multiplier),
                out_channels=get_channels(96, width_multiplier),
                stride=1,
                expansion_factor=6,
                repeats=3,
            ),
            self._get_bottlenecks(
                in_channels=get_channels(96, width_multiplier),
                out_channels=get_channels(160, width_multiplier),
                stride=2,
                expansion_factor=6,
                repeats=3,
            ),
            self._get_bottlenecks(
                in_channels=get_channels(160, width_multiplier),
                out_channels=get_channels(320, width_multiplier),
                stride=1,
                expansion_factor=6,
                repeats=1,
            ),
        )

        self.final_conv = nn.Conv2d(
            get_channels(320, width_multiplier),
            get_channels(1280, width_multiplier) if width_multiplier > 1.0 else 1280,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.final_bn = nn.BatchNorm2d(
            get_channels(1280, width_multiplier) if width_multiplier > 1.0 else 1280
        )

        self.relu6 = nn.ReLU6(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            get_channels(1280, width_multiplier) if width_multiplier > 1.0 else 1280, num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu6(x)

        x = self.bottlenecks(x)

        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.relu6(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _get_bottlenecks(
        self, in_channels: int, out_channels: int, stride: int, expansion_factor: int, repeats: int
    ) -> nn.Sequential:
        layers = []
        for _ in range(repeats):
            layers.append(
                InvertedResidualBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion_factor=expansion_factor,
                )
            )
            in_channels = out_channels
            stride = 1  # Only the first block in the sequence can have a stride > 1
        return nn.Sequential(*layers)


if __name__ == "__main__":
    from thop import profile

    model = MobileNet(input_channels=3, num_classes=1000, width_multiplier=1.0)

    x = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(x,), verbose=False)
    print("MobileNet V1")
    print(f"  MACs: {macs:,}")
    print(f"  Params: {params:,}")

    modelv2 = MobileNetV2(input_channels=3, num_classes=1000, width_multiplier=1.0)
    x = torch.randn(1, 3, 224, 224)
    macs, params = profile(modelv2, inputs=(x,), verbose=False)
    print("MobileNet V2")
    print(f"  MACs: {macs:,}")
    print(f"  Params: {params:,}")
