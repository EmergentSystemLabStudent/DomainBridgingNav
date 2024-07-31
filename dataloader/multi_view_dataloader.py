from __future__ import annotations

from typing import Callable

import random
from tqdm import tqdm 

import torch 
from torch import Tensor
from torch.utils.data import DataLoader, Dataset 

from torchvision import transforms 
from torchvision.datasets import ImageFolder

class MultiViewDataset(Dataset) :
    
    _augmentations: Callable
    _base_dataset: ImageFolder
    
    def __init__(self, base_dataset: ImageFolder, base_augmentation: Callable) -> None:
        
        super(MultiViewDataset, self).__init__()
        
        self._augmentations = base_augmentation
        self._base_dataset = base_dataset
        self._classes = self._base_dataset.classes
        
    def __len__(self) -> int:
        return len(self._base_dataset)
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        
        image, label = self._base_dataset[index]
        same_object_images = [idx for idx, target_label in enumerate(self._base_dataset.targets) if target_label == label]
        other_view, _ = self._base_dataset[random.choice(same_object_images)]
        x0, x1 = self._augmentations(image), self._augmentations(other_view)
        
        return x0, x1, label
    
    @property
    def classes(self) -> list[str]:
        return self._classes