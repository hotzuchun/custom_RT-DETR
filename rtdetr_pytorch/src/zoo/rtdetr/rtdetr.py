"""
Enhanced RT-DETR Implementation
Based on the original RT-DETR project by lyuwenyu

Original Repository: https://github.com/lyuwenyu/RT-DETR
Original Authors: Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, 
                  Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen
Original License: Apache License 2.0

This is an enhanced implementation with improvements and modifications
while maintaining compatibility with the original RT-DETR architecture.
"""

"""by lyuwenyu

"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
