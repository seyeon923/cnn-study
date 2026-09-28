import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2 if downsample else 1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.identity = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.identity is not None:
            identity = self.identity(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = ResidualBlock(64, 64)
        self.conv2_2 = ResidualBlock(64, 64)

        self.conv3_1 = ResidualBlock(64, 128, downsample=True)
        self.conv3_2 = ResidualBlock(128, 128)

        self.conv4_1 = ResidualBlock(128, 256, downsample=True)
        self.conv4_2 = ResidualBlock(256, 256)

        self.conv5_1 = ResidualBlock(256, 512, downsample=True)
        self.conv5_2 = ResidualBlock(512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNet34(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = ResidualBlock(64, 64)
        self.conv2_2 = ResidualBlock(64, 64)
        self.conv2_3 = ResidualBlock(64, 64)

        self.conv3_1 = ResidualBlock(64, 128, downsample=True)
        self.conv3_2 = ResidualBlock(128, 128)
        self.conv3_3 = ResidualBlock(128, 128)
        self.conv3_4 = ResidualBlock(128, 128)

        self.conv4_1 = ResidualBlock(128, 256, downsample=True)
        self.conv4_2 = ResidualBlock(256, 256)
        self.conv4_3 = ResidualBlock(256, 256)
        self.conv4_4 = ResidualBlock(256, 256)
        self.conv4_5 = ResidualBlock(256, 256)
        self.conv4_6 = ResidualBlock(256, 256)

        self.conv5_1 = ResidualBlock(256, 512, downsample=True)
        self.conv5_2 = ResidualBlock(512, 512)
        self.conv5_3 = ResidualBlock(512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        downsample: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=2 if downsample else 1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        if downsample:
            self.identity = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=2 if downsample else 1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        elif in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.identity = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity is not None:
            identity = self.identity(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = Bottleneck(64, 64, 256)
        self.conv2_2 = Bottleneck(256, 64, 256)
        self.conv2_3 = Bottleneck(256, 64, 256)

        self.conv3_1 = Bottleneck(256, 128, 512, downsample=True)
        self.conv3_2 = Bottleneck(512, 128, 512)
        self.conv3_3 = Bottleneck(512, 128, 512)
        self.conv3_4 = Bottleneck(512, 128, 512)

        self.conv4_1 = Bottleneck(512, 256, 1024, downsample=True)
        self.conv4_2 = Bottleneck(1024, 256, 1024)
        self.conv4_3 = Bottleneck(1024, 256, 1024)
        self.conv4_4 = Bottleneck(1024, 256, 1024)
        self.conv4_5 = Bottleneck(1024, 256, 1024)
        self.conv4_6 = Bottleneck(1024, 256, 1024)

        self.conv5_1 = Bottleneck(1024, 512, 2048, downsample=True)
        self.conv5_2 = Bottleneck(2048, 512, 2048)
        self.conv5_3 = Bottleneck(2048, 512, 2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from thop import profile

    resnet18 = ResNet18(input_channels=3, num_classes=1000)

    x = torch.randn(1, 3, 224, 224)
    y = resnet18(x)

    mac, params = profile(resnet18, inputs=(x,), verbose=False)

    print("ResNet-18")
    print(f"    Input: {x.shape}")
    print(f"    Output: {y.shape}")
    print(f"    MACs: {mac:,}")
    print(f"    Params: {params:,}")

    resnet34 = ResNet34(input_channels=3, num_classes=1000)
    y = resnet34(x)

    mac, params = profile(resnet34, inputs=(x,), verbose=False)

    print("ResNet-34")
    print(f"    Input: {x.shape}")
    print(f"    Output: {y.shape}")
    print(f"    MACs: {mac:,}")
    print(f"    Params: {params:,}")

    resnet50 = ResNet50(input_channels=3, num_classes=1000)
    y = resnet50(x)

    mac, params = profile(resnet50, inputs=(x,), verbose=False)

    print("ResNet-50")
    print(f"    Input: {x.shape}")
    print(f"    Output: {y.shape}")
    print(f"    MACs: {mac:,}")
    print(f"    Params: {params:,}")
