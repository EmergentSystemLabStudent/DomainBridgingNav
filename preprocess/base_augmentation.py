from __future__ import annotations

import sys 
from pathlib import Path

from torchvision import transforms

sys.path.append(str(Path(__file__).parent.absolute()))
from modules.gaussian_blur import GaussianBlur
from modules.two_crops_transform import TwoCropsTransform

def get_train_transform() -> TwoCropsTransform:
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)), 
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8), 
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(), 
        normalize
    ]
    
    return TwoCropsTransform(train_transform)

def get_eval_transform() -> list:
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    eval_transform = [
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        normalize
    ]
    
    return eval_transform

def get_label_transform() -> list:
    
    label_img_transform = [
        transforms.Resize(50),
        transforms.ToTensor(),
    ]
    
    return label_img_transform