import torch
from torch import nn


def make_activation(activation: str = "relu") -> nn.Module:
    activation = activation.lower()

    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Not supported activation: {activation}")


class Classifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        output_classes: int,
        hidden_features: int | list[int] = 1024,
        num_hidden_layers: int = 1,
        expected_feature_size: int | tuple[int, int] = (7, 7),
        classifier_type: str = "gap_linear",
        activation: str = "relu",
    ):
        super().__init__()

        if hasattr(hidden_features, "__iter__"):
            assert len(hidden_features) == num_hidden_layers
        else:
            hidden_features = [hidden_features for _ in range(num_hidden_layers)]

        classifier_type = classifier_type.lower()

        if classifier_type == "gap_linear":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_features, output_classes)
            )

        elif classifier_type == "gap_mlp":
            layers = []

            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())

            for out_features in hidden_features:
                layers.append(nn.Linear(in_features, out_features))
                layers.append(make_activation(activation))
                layers.append(nn.Dropout(p=0.5))

                in_features = out_features

            layers.append(nn.Linear(in_features, output_classes))

            self.classifier = nn.Sequential(*layers)

        elif classifier_type == "conv_dense":
            layers = []

            ksize = expected_feature_size

            for out_features in hidden_features:
                layers.append(nn.Conv2d(in_features, out_features, ksize))
                layers.append(make_activation(activation))
                layers.append(nn.Dropout(p=0.5))

                in_features = out_features
                ksize = 1

            layers.append(nn.Conv2d(in_features, output_classes, 1))

            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())

            self.classifier = nn.Sequential(*layers)

        else:
            raise ValueError(
                f"Invalid classifier_type '{classifier_type}'. "
                "Available classifer_type: ['gap_linear', 'gap_mlp', 'conv_dense']"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
