简体中文 | [English](README.md)

# custom_RT-DETR

这是基于 lyuwenyu 原始 RT-DETR 项目的改进。

## 📋 重要声明

**这是基于 lyuwenyu 原始 RT-DETR 项目的改进。**

- **原始仓库**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- **原始论文**: 
  - [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
  - [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)
- **许可证**: Apache 2.0 (继承自原始项目)

本实现包含改进和修改，同时保持与原始 RT-DETR 架构的兼容性。

## 🚀 本实现的改进

- 更方便的个性化设置
- 增加了新的骨干网络: Swin_Transformer, Convnext

## 📍 实现详情

本仓库包含 RT-DETR 的增强 PyTorch 实现，具有以下特性：

- **骨干网络**: ResNet、ConvNeXt（通过 timm）、Swin Transformer（通过 timm）变体
- **训练支持**: 单GPU和多GPU训练
- **导出支持**: ONNX、TensorRT 部署
- **自定义数据集支持**: 易于适配自定义数据集

## 🦄 性能表现 (基于原始论文)

| 模型 | 输入尺寸 | 数据集 | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | 参数量(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |
|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|
| RT-DETR-R18 | 640 | COCO | 46.5 | 63.8 | 20 | 60 | 217 |
| RT-DETR-R34 | 640 | COCO | 48.9 | 66.8 | 31 | 92 | 161 |
| RT-DETR-R50 | 640 | COCO | 53.1 | 71.3 | 42 | 136 | 108 |
| RT-DETR-R101 | 640 | COCO | 54.3 | 72.7 | 76 | 259 | 74 |

## 🏃‍♂️ 快速开始

### 安装
```bash
pip install -r rtdetr_pytorch/requirements.txt
```

**注意**: 本实现需要 `timm>=0.9.0` 来支持 Swin Transformer 和 ConvNeXt 骨干网络。

### 训练
```bash
# 单GPU训练
export CUDA_VISIBLE_DEVICES=0
python rtdetr_pytorch/tools/train.py -c rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml

# 多GPU训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 rtdetr_pytorch/tools/train.py -c rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

### 评估
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 rtdetr_pytorch/tools/train.py -c rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

### 导出ONNX
```bash
python rtdetr_pytorch/tools/export_onnx.py -c rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check
```

## 📚 引用

如果您在研究中使用了本实现，请引用原始论文：

```bibtex
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Yian Zhao and Wenyu Lv and Shangliang Xu and Jinman Wei and Guanzhong Wang and Qingqing Dang and Yi Liu and Jie Chen},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer}, 
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140}, 
}
```

## 🙏 致谢

- **原始作者**: Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen
- **原始仓库**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- **许可证**: 本项目基于原始 RT-DETR 实现，继承 Apache 2.0 许可证。

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。本实现基于 lyuwenyu 的原始 RT-DETR 项目，该项目同样采用 Apache 2.0 许可证。

<details>
<summary>Fig</summary>

<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/77494834/0ede1dc1-a854-43b6-9986-cf9090f11a61" width=500 >
</div>

</details>


## 最新动态
- 发布RT-DETRv2系列模型
- 发布RT-DETR-R50, RT-DETR-R101模型
- 发布RT-DETR-R50-m模型（scale模型的范例）
- 发布RT-DETR-R34, RT-DETR-R18模型
- 发布RT-DETR-L, RT-DETR-X模型


## 代码仓库
- 🔥 RT-DETRv2
  - paddle: [code&weight](./rtdetrv2_paddle/)
  - pytorch: [code&weight](./rtdetrv2_pytorch/)
- 🔥 RT-DETR 
  - paddle: [code&weight](./rtdetr_paddle)
  - pytorch: [code&weight](./rtdetr_pytorch)


## 简介
<!-- We propose a **R**eal-**T**ime **DE**tection **TR**ansformer (RT-DETR), the first real-time end-to-end object detector to our best knowledge. Specifically, we design an efficient hybrid encoder to efficiently process multi-scale features by decoupling the intra-scale interaction and cross-scale fusion, and propose IoU-aware query selection to improve the initialization of object queries. In addition, our proposed detector supports flexibly adjustment of the inference speed by using different decoder layers without the need for retraining, which facilitates the practical application of real-time object detectors. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS.  -->
RT-DETR是第一个实时端到端目标检测器。具体而言，我们设计了一个高效的混合编码器，通过解耦尺度内交互和跨尺度融合来高效处理多尺度特征，并提出了IoU感知的查询选择机制，以优化解码器查询的初始化。此外，RT-DETR支持通过使用不同的解码器层来灵活调整推理速度，而不需要重新训练，这有助于实时目标检测器的实际应用。RT-DETR-R50在COCO val2017上实现了53.1%的AP，在T4 GPU上实现了108FPS，RT-DETR-R101实现了54.3%的AP和74FPS，在速度和精度方面都优于相同规模的所有YOLO检测器。使用Objects365预训练之后, RT-DETR-R50 和 RT-DETR-R101 分别实现了 55.3% 和 56.2% AP的精度.
若要了解更多细节，请参考我们的论文[paper](https://arxiv.org/abs/2304.08069).

<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/77494834/c211a164-ddce-4084-8b71-fb73f29f363b" width=500 >
</div>

## 引用RT-DETR
如果需要在你的研究中使用RT-DETR，请通过以下方式引用我们的论文：
```
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Yian Zhao and Wenyu Lv and Shangliang Xu and Jinman Wei and Guanzhong Wang and Qingqing Dang and Yi Liu and Jie Chen},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer}, 
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140}, 
}
```
