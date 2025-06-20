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
import torchvision



def format_target(targets):
    '''
    Args:
        targets (List[Dict]),
    Return: 
        tensor (Tensor), [im_id, label, bbox,]
    '''
    outputs = []
    for i, tgt in enumerate(targets):
        boxes =  torchvision.ops.box_convert(tgt['boxes'], in_fmt='xyxy', out_fmt='cxcywh') 
        labels = tgt['labels'].reshape(-1, 1)
        im_ids = torch.ones_like(labels) * i
        outputs.append(torch.cat([im_ids, labels, boxes], dim=1))

    return torch.cat(outputs, dim=0)
