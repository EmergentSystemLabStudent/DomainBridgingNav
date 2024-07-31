from __future__ import annotations

import sys 
from pathlib import Path

from torchvision import transforms
from torchvision.transforms import Compose

sys.path.append(str(Path(__file__).parent.absolute()))
from modules.gaussian_blur import GaussianBlur

def get_train_transform() -> Compose :
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = [
        transforms.Resize((256, 256)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8), 
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(), 
        normalize
    ]
    
    return Compose(train_transform)