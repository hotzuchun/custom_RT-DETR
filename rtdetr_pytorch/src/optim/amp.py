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
import torch.cuda.amp as amp


from src.core import register
import src.misc.dist as dist 


__all__ = ['GradScaler']

GradScaler = register(amp.grad_scaler.GradScaler)
