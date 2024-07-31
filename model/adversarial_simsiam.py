from __future__ import annotations

import sys
from pathlib import Path 
from typing import Tuple

import torch 
from torch import Tensor
from torch import nn
from torchvision.models.resnet import ResNet

sys.path.append(str(Path(__file__).parent.absolute()))
from backbone.resnet import ResNet18, ResNet50

class GRLFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_forward: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)
        return input_forward
 
    @staticmethod
    def backward(ctx, grad_backward: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, = ctx.saved_tensors
        return scale * -grad_backward, None
    
class GRL(nn.Module):
 
    def __init__(self, scale: float):
        super(GRL, self).__init__()
        self.scale = torch.tensor(scale)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GRLFunction.apply(x, self.scale)

class SimSiam(nn.Module) :
    
    _backbone: ResNet
    
    def __init__(
        self, 
        backbone: str="res50", 
        backbone_dim: int=2048, 
        prediction_dim: int=512, 
        num_classes: int=10,
        imagenet_pretrain_path: str=None,
        freeze_partial_param: bool=False
    ) -> None:
        
        super(SimSiam, self).__init__()
        
        if backbone == "res50" :
            self._backbone = ResNet50(backbone_dim)
            
        elif backbone == "res18" :
            self._backbone = ResNet18(backbone_dim)
            
        else :
            raise f"{backbone} is not supported"
        
        hidden_dim = self._backbone.fc.weight.shape[1]
        
        self._projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(hidden_dim, hidden_dim, bias=False),
                                   nn.BatchNorm1d(hidden_dim), 
                                   nn.ReLU(inplace=True),
                                   self._backbone.fc, 
                                   nn.BatchNorm1d(backbone_dim, affine=False))
        
        self._backbone.fc = nn.Identity()
        self._projector[6].bias.requires_grad = False
        
        self._predictor = nn.Sequential(
                           nn.Linear(backbone_dim, prediction_dim, bias=False),
                           nn.BatchNorm1d(prediction_dim),
                           nn.ReLU(inplace=True), # hidden layer
                           nn.Linear(prediction_dim, backbone_dim)) # output layer
        
        self._classifier = nn.Linear(hidden_dim, num_classes)
        self._classifier.weight.data.normal_(mean=0.0, std=0.01)
        self._classifier.bias.data.zero_()
        
        self._grl = GRL(scale=1.0)
        
        self._domain_discriminator = nn.Linear(hidden_dim, 2)
        self._domain_discriminator.weight.data.normal_(mean=0.0, std=0.01)
        self._domain_discriminator.bias.data.zero_()
        
        if imagenet_pretrain_path is not None:
            self.load_imagenet(imagenet_pretrain_path)
            
        if freeze_partial_param:
            for name, param in self.named_parameters():
                if name.startswith("_backbone") and "layer4" not in name:
                    param.requires_grad = False
                # if name.startswith("_projector") or name.startswith("_predictor"):
                #     param.requires_grad = False
                    
                print(f"{name} : grad {param.requires_grad}")
            
        
    def forward(self, x1:Tensor, x2:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        
        z1 = self._backbone(x1) # NxC
        h1 = self._projector(z1) # NxC
        
        z2 = self._backbone(x2) # NxC
        h2 = self._projector(z2) # NxC

        p1 = self._predictor(h1) # NxC
        p2 = self._predictor(h2) # NxC
        
        logits1 = self._classifier(z1)
        logits2 = self._classifier(z2)
        
        z1_reverse = self._grl(z1)
        z2_reverse = self._grl(z2)
        
        domain_logits1 = self._domain_discriminator(z1_reverse)
        domain_logits2 = self._domain_discriminator(z2_reverse)
        
        return p1, p2, h1.detach(), h2.detach(), logits1, logits2.detach(), domain_logits1, domain_logits2.detach()
    
    def get_embedding(self, image:Tensor, auto_cast: bool = True) -> Tensor:
        
        if auto_cast: 
            with torch.cuda.amp.autocast():
                embedding = self._backbone(image)
                
        else:
            embedding = self._backbone(image)
            
        return embedding.detach().cpu().numpy()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        backbone_state_dict = {}
        projector_state_dict = {}
        predictor_state_dict = {}
        classifier_state_dict = {}
        discriminator_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith("_backbone"):
                backbone_state_dict[key.replace("_backbone.", "")] = value
            elif key.startswith("_projector"):
                projector_state_dict[key.replace("_projector.", "")] = value
            elif key.startswith("_predictor"):
                predictor_state_dict[key.replace("_predictor.", "")] = value
            elif key.startswith("_classifier"):
                classifier_state_dict[key.replace("_classifier.", "")] = value
            elif key.startswith("_domain_discriminator"):
                discriminator_state_dict[key.replace("_domain_discriminator.", "")] = value
            else:
                raise ValueError(f"unexpected key: {key}")
            
        backbone_msg = self._backbone.load_state_dict(backbone_state_dict, strict=True)
        projector_msg = self._projector.load_state_dict(projector_state_dict, strict=True)
        predictor_msg = self._predictor.load_state_dict(predictor_state_dict, strict=True)
        print(f"Backbone Msg: {backbone_msg}")
        print(f"Projector Msg: {projector_msg}")
        print(f"Predictor Msg: {predictor_msg}")
        
    def load_imagenet(self, imagenet_path: str) -> None:
        
        checkpoint = torch.load(imagenet_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        encoder_state_dict = {}
        projector_state_dict = {}
        predictor_state_dict = {}
        
        
        for key, value in state_dict.items():
            if key.startswith("module.encoder") and not key.startswith("module.encoder.fc"):
                encoder_state_dict[key.replace("module.encoder.", "")] = value
            elif key.startswith("module.encoder.fc"):
                projector_state_dict[key.replace("module.encoder.fc.", "")] = value
            elif key.startswith("module.predictor"):
                predictor_state_dict[key.replace("module.predictor.", "")] = value
            else:
                raise ValueError(f"unexpected key: {key}")
            
        encoder_msg = self._backbone.load_state_dict(encoder_state_dict, strict=True)
        projector_msg = self._projector.load_state_dict(projector_state_dict, strict=True)
        predictor_msg = self._predictor.load_state_dict(predictor_state_dict, strict=True)
        print(f"Encoder Msg: {encoder_msg}")
        print(f"Projector Msg: {projector_msg}")
        print(f"Predictor Msg: {predictor_msg}")
