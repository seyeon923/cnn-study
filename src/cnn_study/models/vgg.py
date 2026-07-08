import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = False):
        super().__init__()

        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.conv_block(x)


class ConvBlockLayers(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, use_bn: bool = False):
        super().__init__()

        layers = []

        in_ch = in_channels
        for _ in range(num_layers):
            layers.append(ConvBlock(in_ch, out_channels, use_bn=use_bn))
            in_ch = out_channels

        layers.append(nn.MaxPool2d(2, 2))

        self.conv_block_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.conv_block_layers(x)


class Classifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        output_classes: int,
        hidden_features: int = 4096,
        num_hidden_layers: int = 2,
        expected_feature_size: int | tuple[int, int] = (7, 7),
        classifier_type: str = "vgg_dense",
    ):
        super().__init__()

        classifier_type = classifier_type.lower()
        if classifier_type == "vgg_dense":
            assert num_hidden_layers >= 1

            layers = []

            layers.append(nn.Conv2d(in_features, hidden_features, expected_feature_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.5))

            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Conv2d(hidden_features, hidden_features, 1))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=0.5))

            layers.append(nn.Conv2d(hidden_features, output_classes, 1))

            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())

            self.classifier = nn.Sequential(*layers)
        elif classifier_type == "gap_mlp":
            layers = []

            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())

            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(in_features, hidden_features))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=0.5))

                in_features = hidden_features

            layers.append(nn.Linear(in_features, output_classes))

            self.classifier = nn.Sequential(*layers)
        elif classifier_type == "gap_linear":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_features, output_classes)
            )
        else:
            raise ValueError(
                f"Invalid classifier_type '{classifier_type}', should be one of 'vgg_dense', 'gap_mlp', or 'gap_linear'",
            )

    def forward(self, x: torch.Tensor):
        return self.classifier(x)


class VGG16(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_classes: int = 1000,
        use_bn: bool = False,
        classifier_type: str = "vgg_dense",
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_classes = output_classes
        self.classifier_type = classifier_type

        self.features = nn.Sequential(
            ConvBlockLayers(input_channels, 64, 2, use_bn=use_bn),  # 224 x 224 => 112 x 112
            ConvBlockLayers(64, 128, 2, use_bn=use_bn),  # 112 x 112 => 56 x 56
            ConvBlockLayers(128, 256, 3, use_bn=use_bn),  # 56 x 56 => 28 x 28
            ConvBlockLayers(256, 512, 3, use_bn=use_bn),  # 28 x 28 => 14 x 14
            ConvBlockLayers(512, 512, 3, use_bn=use_bn),  # 14 x 14 => 7 x 7
        )

        self.classifier = Classifier(
            512,
            output_classes,
            hidden_features=4096,
            num_hidden_layers=2,
            expected_feature_size=7,
            classifier_type=classifier_type,
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        return self.classifier(x)


class VGG19(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_classes: int = 1000,
        use_bn: bool = False,
        classifier_type: str = "vgg_dense",
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_classes = output_classes
        self.classifier_type = classifier_type

        self.features = nn.Sequential(
            ConvBlockLayers(input_channels, 64, 2, use_bn=use_bn),
            ConvBlockLayers(64, 128, 2, use_bn=use_bn),
            ConvBlockLayers(128, 256, 4, use_bn=use_bn),
            ConvBlockLayers(256, 512, 4, use_bn=use_bn),
            ConvBlockLayers(512, 512, 4, use_bn=use_bn),
        )

        self.classifier = Classifier(
            512,
            output_classes,
            hidden_features=4096,
            num_hidden_layers=2,
            expected_feature_size=7,
            classifier_type=classifier_type,
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    classifier_types = ["vgg_dense", "gap_mlp", "gap_linear"]

    for classifier_type in classifier_types:
        output_classes = 100
        vgg16 = VGG16(output_classes=output_classes, classifier_type=classifier_type)
        x = torch.randn(1, 3, 224, 224)
        y = vgg16(x)

        assert y.shape == (1, output_classes)

        print(f"Input: {x.shape}")
        print(f"Output: {y.shape}")
        print()

        x = torch.randn(10, 3, 384, 384)
        y = vgg16(x)

        assert y.shape == (10, output_classes)

        print(f"Input: {x.shape}")
        print(f"Output: {y.shape}")
        print()

        vgg19 = VGG19(output_classes=output_classes, classifier_type=classifier_type)
        x = torch.randn(1, 3, 224, 224)
        y = vgg19(x)

        assert y.shape == (1, output_classes)

        print(f"Input: {x.shape}")
        print(f"Output: {y.shape}")
        print()

        x = torch.randn(10, 3, 384, 384)
        y = vgg19(x)

        assert y.shape == (10, output_classes)

        print(f"Input: {x.shape}")
        print(f"Output: {y.shape}")
        print()
