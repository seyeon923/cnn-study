from typing import Any

from torch import nn

from .alexnet import AlexNet
from .lenet import LeNet5

MODEL_REGISTRY: dict[str, type[nn.Module]] = {"lenet": LeNet5, "alexnet": AlexNet}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
