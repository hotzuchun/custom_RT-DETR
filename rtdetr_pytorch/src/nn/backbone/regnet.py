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

import torch
import torch.nn as nn 
from transformers import RegNetModel


from src.core import register

__all__ = ['RegNet']

@register
class RegNet(nn.Module):
    def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
        super(RegNet, self).__init__()  
        self.model = RegNetModel.from_pretrained("facebook/regnet-y-040")
        self.return_idx = return_idx


    def forward(self, x):
        
        outputs = self.model(x, output_hidden_states = True)
        x = outputs.hidden_states[2:5]

        return x