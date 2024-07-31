import math 

import torch

class CosineScheduler:
    
    @staticmethod
    def adsust_lr(optimizer: torch.optim, epoch: int, epochs: int,  init_lr: float) -> None:
        """Decay the learning rate based on schedule"""
        
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr
        