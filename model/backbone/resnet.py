from torch import nn
from torchvision.models.resnet import resnet18, resnet50

def ResNet18(num_classes: int=2048):
    return resnet18(pretrained=False, num_classes=num_classes, zero_init_residual=True)

def ResNet50(num_classes: int=2048):
    return resnet50(pretrained=False, num_classes=num_classes, zero_init_residual=True)

