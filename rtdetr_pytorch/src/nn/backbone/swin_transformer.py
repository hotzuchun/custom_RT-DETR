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

__all__ = ['SwinTransformer']

SWIN_CFG = {
    'tiny': {
        'model_name': 'swin_tiny_patch4_window7_224',
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'out_channels': [96, 192, 384],
    },
    'small': {
        'model_name': 'swin_small_patch4_window7_224',
        'embed_dim': 96,
        'depths': [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
        'out_channels': [96, 192, 384],
    },
    'base': {
        'model_name': 'swin_base_patch4_window7_224',
        'embed_dim': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'out_channels': [128, 256, 512],
    },
    'large': {
        'model_name': 'swin_large_patch4_window7_224',
        'embed_dim': 192,
        'depths': [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48],
        'out_channels': [192, 384, 768],
    },
}

@register
class SwinTransformer(nn.Module):
    def __init__(self, 
                 arch='tiny',
                 img_size=224, 
                 patch_size=4, 
                 in_chans=3, 
                 embed_dim=None,
                 depths=None,
                 num_heads=None,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 pretrained=False,
                 return_idx=[0, 1, 2]):
        super().__init__()
        
        # 选择配置
        assert arch in SWIN_CFG, f"Unsupported arch: {arch}"
        cfg = SWIN_CFG[arch]
        self.arch = arch
        self.model_name = cfg['model_name']
        self.embed_dim = embed_dim if embed_dim is not None else cfg['embed_dim']
        self.depths = depths if depths is not None else cfg['depths']
        self.num_heads = num_heads if num_heads is not None else cfg['num_heads']
        self.out_channels = cfg['out_channels']
        self.out_strides = [8, 16, 32]  # 保持与PResNet一致

        # 验证输入参数
        if img_size % patch_size != 0:
            raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}")
        if max(return_idx) >= len(self.depths):
            raise ValueError(f"return_idx {return_idx} exceeds number of stages {len(self.depths)}")

        # 创建timm模型
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            drop_block_rate=None,
            global_pool='',
            num_classes_timm=0,
            features_only=True,
            out_indices=return_idx,
        )
        self.return_idx = return_idx

        # 计算每个stage的特征图尺寸
        self.img_size = img_size
        self.patch_size = patch_size
        self.stage_sizes = []
        size = img_size // patch_size
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
            B, H, W, C = feat.shape
            feat = feat.permute(0, 3, 1, 2).contiguous()
            # 强制resize到目标尺寸
            feat = torch.nn.functional.interpolate(feat, size=(target_sizes[i], target_sizes[i]), mode='bilinear', align_corners=False)
            outs.append(feat)
        return outs
