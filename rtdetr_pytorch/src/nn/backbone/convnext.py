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

'''by HO TZU CHUN

'''
import torch
import torch.nn as nn
import timm

from src.core import register

__all__ = ['ConvNeXt']

CONVNEXT_CFG = {
    'tiny': {
        'model_name': 'convnext_tiny',
        'depths': [3, 3, 9, 3],
        'dims': [96, 192, 384, 768],
        'out_channels': [96, 192, 384],
    },
    'small': {
        'model_name': 'convnext_small',
        'depths': [3, 3, 27, 3],
        'dims': [96, 192, 384, 768],
        'out_channels': [96, 192, 384],
    },
    'base': {
        'model_name': 'convnext_base',
        'depths': [3, 3, 27, 3],
        'dims': [128, 256, 512, 1024],
        'out_channels': [128, 256, 512],
    },
    'large': {
        'model_name': 'convnext_large',
        'depths': [3, 3, 27, 3],
        'dims': [192, 384, 768, 1536],
        'out_channels': [192, 384, 768],
    },
}

@register
class ConvNeXt(nn.Module):
    def __init__(self, 
                 arch='tiny',
                 img_size=224, 
                 in_chans=3, 
                 depths=None,
                 dims=None,
                 drop_path_rate=0.,
                 pretrained=False,
                 return_idx=[0, 1, 2]):
        super().__init__()
        
        # 选择配置
        assert arch in CONVNEXT_CFG, f"Unsupported arch: {arch}"
        cfg = CONVNEXT_CFG[arch]
        self.arch = arch
        self.model_name = cfg['model_name']
        self.depths = depths if depths is not None else cfg['depths']
        self.dims = dims if dims is not None else cfg['dims']
        self.out_channels = cfg['out_channels']
        self.out_strides = [8, 16, 32]  # 保持与PResNet一致
        self.img_size = img_size

        # 验证输入参数
        if max(return_idx) >= len(self.depths):
            raise ValueError(f"return_idx {return_idx} exceeds number of stages {len(self.depths)}")

        # 创建timm模型
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            features_only=True,
            out_indices=return_idx,
        )
        self.return_idx = return_idx

        # 计算每个stage的特征图尺寸
        self.stage_sizes = []
        size = img_size
        for _ in range(len(self.depths)):
            self.stage_sizes.append(size)
            size = size // 2

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"Input image size {H}x{W} does not match expected size {self.img_size}x{self.img_size}")
        features = self.backbone(x)
        outs = []
        # 计算目标尺寸（如[80, 40, 20]）
        target_sizes = [self.img_size // s for s in self.out_strides]
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            # 强制resize到目标尺寸
            feat = torch.nn.functional.interpolate(feat, size=(target_sizes[i], target_sizes[i]), mode='bilinear', align_corners=False)
            outs.append(feat)
        return outs 