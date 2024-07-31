from __future__ import annotations
from typing import Callable

from torch import Tensor
from torchvision import transforms

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    _base_transform: Callable

    def __init__(self, base_transform: list):
        self._base_transform = transforms.Compose(base_transform)

    def __call__(self, x: Tensor) -> list[Tensor, Tensor]:
        q = self._base_transform(x)
        k = self._base_transform(x)
        return [q, k] 