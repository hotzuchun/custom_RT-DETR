# custom_RT-DETR 

This document describes the specific enhancements and modifications made to the original RT-DETR implementation by lyuwenyu.

## Overview

This is a modification of RT-DETR, you can train in your custom dataset and there are more backbone options.

## Key Enhancements

### 1. Custom dataset

#### Convinent Customization
- You can set your custom dataset more convinent.

### 2. Additional Backbone Networks

#### ConvNeXt Support
- Added ConvNeXt backbone variants (tiny, small, base, large) based on timm library
- Optimized configurations for ConvNeXt-based RT-DETR models
- Performance benchmarks and comparison with ResNet variants

#### Swin Transformer Support
- Add Swin Transformer backbone variants (tiny, small, base, large) based on timm library
- Optimized configurations for different Swin variants
- Improved feature extraction pipeline for transformer-based backbones

### 3. Dependencies

#### New Dependencies
- **timm>=0.9.0**: Required for Swin Transformer and ConvNeXt backbone networks
- All original dependencies are preserved for compatibility

## Acknowledgments

This modification builds upon the excellent work of the original RT-DETR authors:
- **Original Authors**: Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen
- **Original Repository**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- **Original Papers**: 
  - [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
  - [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)

All original copyright notices and licenses are preserved and respected. 