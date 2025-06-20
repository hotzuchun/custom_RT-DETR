# custom_RT-DETR

本文档详细描述了基于 lyuwenyu 原始 RT-DETR 项目的增强实现的具体改进和修改。

## 概要

这是对 RT-DETR 的改进，您可以在自定义数据集上进行训练，并且有更多的骨干网络选项。

## 主要改进

### 1. 自定义数据集
- 根据注释设置你的自定义数据集

### 2. 额外的骨干网络

#### 新增ConvNeXt 支持
- 添加 ConvNeXt 骨干网络变体（tiny、small、base、large），基于 timm 库
- ConvNeXt 基础 RT-DETR 模型的优化配置
- 与 ResNet 变体的性能基准和比较

#### 新增Swin Transformer 支持
- 增强的 Swin Transformer 骨干网络支持，基于 timm 库
- 针对不同 Swin 变体的优化配置
- 基于 Transformer 骨干网络的改进特征提取流程

### 3. 依赖项

#### 新增依赖
- **timm>=0.9.0**: Swin Transformer 和 ConvNeXt 骨干网络所需
- 保留所有原始依赖项以确保兼容性

## 致谢

此增强实现建立在原始 RT-DETR 作者的优秀工作基础上：
- **原始作者**: Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen
- **原始仓库**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- **原始论文**: 
  - [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
  - [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)

所有原始版权声明和许可证都得到保留和尊重。 