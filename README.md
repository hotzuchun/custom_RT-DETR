English | [简体中文](README_cn.md)


<h2 align="center">RT-DETR: Enhanced Implementation for Real-time Object Detection</h2>
<p align="center">
    <!-- <a href="https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a> -->
    <a href="https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/lyuwenyu/RT-DETR">
    </a>
    <a href="https://github.com/lyuwenyu/RT-DETR/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/lyuwenyu/RT-DETR">
    </a>
    <a href="https://github.com/lyuwenyu/RT-DETR/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/lyuwenyu/RT-DETR?color=pink">
    </a>
    <a href="https://github.com/lyuwenyu/RT-DETR">
        <img alt="issues" src="https://img.shields.io/github/stars/lyuwenyu/RT-DETR">
    </a>
    <a href="https://arxiv.org/abs/2304.08069">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2304.08069-red">
    </a>
    <a href="https://arxiv.org/abs/2407.17140">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2407.17140-red">
    </a>
    <a href="mailto: lyuwenyu@foxmail.com">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>

---

## 📋 Important Notice

**This is an modification based on the original RT-DETR project by lyuwenyu.**

- **Original Repository**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- **Original Papers**: 
  - [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
  - [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)
- **License**: Apache 2.0 (inherited from original project)

This implementation includes improvements and modifications while maintaining compatibility with the original RT-DETR architecture.

## 🚀 What's New in This Implementation

- More convinent customization
- More backbone options

## 📍 Implementation Details

This repository contains an enhanced PyTorch implementation of RT-DETR with the following features:

- **Backbone Networks**: ResNet, ConvNeXt (via timm), Swin Transformer (via timm) variants
- **Training Support**: Single and multi-GPU training
- **Export Support**: ONNX, TensorRT deployment
- **Custom Dataset Support**: Easy adaptation for custom datasets

## 🦄 Performance (Based on Original Paper)

| Model | Input shape | Dataset | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |
|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|
| RT-DETR-R18 | 640 | COCO | 46.5 | 63.8 | 20 | 60 | 217 |
| RT-DETR-R34 | 640 | COCO | 48.9 | 66.8 | 31 | 92 | 161 |
| RT-DETR-R50 | 640 | COCO | 53.1 | 71.3 | 42 | 136 | 108 |
| RT-DETR-R101 | 640 | COCO | 54.3 | 72.7 | 76 | 259 | 74 |

## 🏃‍♂️ Quick Start

### Installation
```bash
pip install -r rtdetr_pytorch/requirements.txt
```

**Note**: This implementation requires `timm>=0.9.0` for Swin Transformer and ConvNeXt backbone networks.

### Training
```bash
# Single GPU training
export CUDA_VISIBLE_DEVICES=0
python rtdetr_pytorch/tools/train.py -c rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml

# Multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 rtdetr_pytorch/tools/train.py -c rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

### Evaluation
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 rtdetr_pytorch/tools/train.py -c rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

### Export ONNX
```bash
python rtdetr_pytorch/tools/export_onnx.py -c rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check
```

## 📚 Citation

If you use this implementation in your research, please cite the original papers:

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

## 🙏 Acknowledgments

- **Original Authors**: Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen
- **Original Repository**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- **License**: This project is based on the original RT-DETR implementation and inherits the Apache 2.0 license.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. This implementation is based on the original RT-DETR project by lyuwenyu, which is also licensed under Apache 2.0.

<details>
<summary>Fig</summary>

<table><tr>
<td><img src=https://github.com/lyuwenyu/RT-DETR/assets/77494834/0ede1dc1-a854-43b6-9986-cf9090f11a61 border=0 width=500></td>
<td><img src=https://github.com/user-attachments/assets/437877e9-1d4f-4d30-85e8-aafacfa0ec56 border=0 width=500></td>
</tr></table>
</details>


## 🚀 Updates
- \[2024.11.28\] Add torch tool for parameters and flops statistics. see [run_profile.py](./rtdetrv2_pytorch/tools/run_profile.py)
- \[2024.10.10\] Add sliced inference support for small object detecion. [#468](https://github.com/lyuwenyu/RT-DETR/pull/468)
- \[2024.09.23\] Add ✅[Regnet and DLA34](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch) for RTDETR.
- \[2024.08.27\] Add hubconf.py file to support torch hub.
- \[2024.08.22\] Improve the performance of ✅ [RT-DETRv2-S](./rtdetrv2_pytorch/) to 48.1 mAP (<font color=green>+1.6</font> compared to RT-DETR-R18).
- \[2024.07.24\] Release ✅ [RT-DETRv2](./rtdetrv2_pytorch/)!
- \[2024.02.27\] Our work has been accepted to CVPR 2024!
- \[2024.01.23\] Fix difference on data augmentation with paper in rtdetr_pytorch [#84](https://github.com/lyuwenyu/RT-DETR/commit/5dc64138e439247b4e707dd6cebfe19d8d77f5b1).
- \[2023.11.07\] Add pytorch ✅ *rtdetr_r34vd* for requests [#107](https://github.com/lyuwenyu/RT-DETR/issues/107), [#114](https://github.com/lyuwenyu/RT-DETR/issues/114).
- \[2023.11.05\] Upgrade the logic of `remap_mscoco_category` to facilitate training of custom datasets, see detils in [*Train custom data*](./rtdetr_pytorch/) part. [#81](https://github.com/lyuwenyu/RT-DETR/commit/95fc522fd7cf26c64ffd2ad0c622c392d29a9ebf).
- \[2023.10.23\] Add [*discussion for deployments*](https://github.com/lyuwenyu/RT-DETR/issues/95), supported onnxruntime, TensorRT, openVINO.
- \[2023.10.12\] Add tuning code for pytorch version, now you can tuning rtdetr based on pretrained weights.
- \[2023.09.19\] Upload ✅ [*pytorch weights*](https://github.com/lyuwenyu/RT-DETR/issues/42) convert from paddle version.
- \[2023.08.24] Release RT-DETR-R18 pretrained models on objects365. *49.2 mAP* and *217 FPS*.
- \[2023.08.22\] Upload ✅ [*rtdetr_pytorch*](./rtdetr_pytorch/) source code. Please enjoy it!
- \[2023.08.15\] Release RT-DETR-R101 pretrained models on objects365. *56.2 mAP* and *74 FPS*.
- \[2023.07.30\] Release RT-DETR-R50 pretrained models on objects365. *55.3 mAP* and *108 FPS*.
- \[2023.07.28\] Fix some bugs, and add some comments. [1](https://github.com/lyuwenyu/RT-DETR/pull/14), [2](https://github.com/lyuwenyu/RT-DETR/commit/3b5cbcf8ae3b907e6b8bb65498a6be7c6736eabc).
- \[2023.07.13\] Upload ✅ [*training logs on coco*](https://github.com/lyuwenyu/RT-DETR/issues/8).
- \[2023.05.17\] Release RT-DETR-R18, RT-DETR-R34, RT-DETR-R50-m（example for scaled).
- \[2023.04.17\] Release RT-DETR-R50, RT-DETR-R101, RT-DETR-L, RT-DETR-X.

## 📣 News
- RTDETR and RTDETRv2 are now available in Hugging Face Transformers. [#413](https://github.com/lyuwenyu/RT-DETR/issues/413), [#549](https://github.com/lyuwenyu/RT-DETR/issues/549)
- RTDETR is now available in [ultralytics/ultralytics](https://docs.ultralytics.com/zh/models/rtdetr/).

## 📍 Implementations
- 🔥 RT-DETRv2
  - paddle: [code&weight](./rtdetrv2_paddle/)
  - pytorch: [code&weight](./rtdetrv2_pytorch/)
- 🔥 RT-DETR 
  - paddle: [code&weight](./rtdetr_paddle)
  - pytorch: [code&weight](./rtdetr_pytorch)


| Model | Input shape | Dataset | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS)
|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|
| RT-DETR-R18 | 640 | COCO | 46.5 | 63.8 | 20 | 60 | 217 |
| RT-DETR-R34 | 640 | COCO | 48.9 | 66.8 | 31 | 92 | 161 |
| RT-DETR-R50-m | 640 | COCO | 51.3 | 69.6 | 36 | 100 | 145 |
| RT-DETR-R50 |  640 | COCO | 53.1 | 71.3 | 42 | 136 | 108 |
| RT-DETR-R101 | 640 | COCO | 54.3 | 72.7 | 76 | 259 | 74 |
| RT-DETR-HGNetv2-L | 640 | COCO | 53.0 | 71.6 | 32 | 110 | 114 |
| RT-DETR-HGNetv2-X | 640 | COCO | 54.8 | 73.1 | 67 | 234 | 74 |
| RT-DETR-R18 | 640 | COCO + Objects365 | **49.2** | **66.6** | 20 | 60 | **217** |
| RT-DETR-R50 | 640 | COCO + Objects365 | **55.3** | **73.4** | 42 | 136 | **108** |
| RT-DETR-R101 | 640 | COCO + Objects365 | **56.2** | **74.6** | 76 | 259 | **74** |
**RT-DETRv2-S** | 640 | COCO  | **48.1** <font color=green>(+1.6)</font> | **65.1** | 20 | 60 | 217 |
**RT-DETRv2-M**<sup>*<sup> | 640 | COCO  | **49.9** <font color=green>(+1.0)</font> | **67.5** | 31 | 92 | 161 |
**RT-DETRv2-M** | 640 | COCO | **51.9** <font color=green>(+0.6)</font> | **69.9** | 36 | 100 | 145 |
**RT-DETRv2-L** | 640 | COCO | **53.4** <font color=green>(+0.3)</font> | **71.6** | 42 | 136 | 108 |
**RT-DETRv2-X** | 640 | COCO | 54.3 | **72.8** <font color=green>(+0.1)</font>  | 76 | 259| 74 |

**Notes:**
- `COCO + Objects365` in the table means finetuned model on COCO using pretrained weights trained on Objects365.


## 🦄 Performance

### 🏕️ Complex Scenarios
<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/77494834/52743892-68c8-4e53-b782-9f89221739e4" width=500 >
</div>

### 🌋 Difficult Conditions
<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/77494834/213cf795-6da6-4261-8549-11947292d3cb" width=500 >
</div>
