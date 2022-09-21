# LSRGAN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Optimizing Generative Adversarial Networks for Image Super Resolution via Latent Space Regularization](https://arxiv.org/pdf/2001.08126.pdf)
.

## Table of contents

- [LSRGAN-PyTorch](#lsrgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train LSRResNet model](#train-lsrresnet-model)
        - [Resume train LSRResNet model](#resume-train-lsrresnet-model)
        - [Train LSRGAN model](#train-lsrgan-model)
        - [Resume train LSRGAN model](#resume-train-lsrgan-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Optimizing Generative Adversarial Networks for Image Super Resolution via Latent Space Regularization](#optimizing-generative-adversarial-networks-for-image-super-resolution-via-latent-space-regularization)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `lsrresnet_config.py` or `lsrgan_config.py` file.

### Test

modify the `lsrgan_config.py`
- line 31: `g_arch_name` change to `lsrgan_x4`.
- line 38: `upscale_factor` change to `4`.
- line 40: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/BSRGAN_x4-DIV2K-6d507222.pth.tar`.
-

```bash
python3 test.py
```

### Train LSRResNet model

modify the `lsrresnet_config.py`
- line 31: `g_arch_name` change to `lsrgan_x4`.
- line 38: `upscale_factor` change to `4`.
- line 40: `mode` change to `train`.
- line 55: `pretrained_model_weights_path` change to `./results/pretrained_models/BSRGAN_x4-DIV2K-6d507222.pth.tar`.

```bash
python3 train_bsrnet.py
```

### Resume train LSRResNet model

modify the `lsrresnet_config.py`
- line 31: `g_arch_name` change to `lsrgan_x4`.
- line 38: `upscale_factor` change to `4`.
- line 40: `mode` change to `train`.
- line 59: `resume` change to `samples/LSRResNet_x4/epoch_xxx.pth.tar`.

```bash
python3 train_bsrnet.py
```

### Train LSRGAN model

modify the `lsrgan_config.py`
- line 32: `g_arch_name` change to `discriminator`.
- line 32: `g_arch_name` change to `lsrgan_x4`.
- line 39: `upscale_factor` change to `4`.
- line 41: `mode` change to `train`.
- line 57: `pretrained_d_model_weights_path` change to `./results/pretrained_models/LSRGAN_x4-DIV2K-e19a5cef.pth.tar`.
- line 58: `pretrained_g_model_weights_path` change to `./results/pretrained_models/LSRGAN_x4-DIV2K-e19a5cef.pth.tar`.

```bash
python3 train_bsrgan.py
```

### Resume train LSRGAN model

modify the `lsrgan_config.py`
- line 32: `g_arch_name` change to `discriminator`.
- line 32: `g_arch_name` change to `lsrgan_x4`.
- line 39: `upscale_factor` change to `4`.
- line 41: `mode` change to `train`.
- line 61: `resume_d` change to `./results/pretrained_models/LSRGAN_x4-DIV2K-e19a5cef.pth.tar`.
- line 62: `resume_g` change to `./results/pretrained_models/LSRGAN_x4-DIV2K-e19a5cef.pth.tar`.


```bash
python3 train_bsrgan.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/2001.08126.pdf](https://arxiv.org/pdf/2001.08126.pdf)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

| Set14 | Scale |  LSRResNet   |    LSRGAN    |
|:-----:|:-----:|:------------:|:------------:|
| PSNR  |   4   | -(**30.87**) | 26.46(**-**) |
| SSIM  |   4   | -(**0.877**) | 0.724(**-**) |
NIQE 5.50

```bash
# Download `LSRGAN_x2-DIV2K-e19a5cef.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input:

<span align="center"><img width="240" height="360" src="figure/comic_lr.png"/></span>

Output:

<span align="center"><img width="240" height="360" src="figure/comic_sr.png"/></span>

```text
Build `lsrgan_x4` model successfully.
Load `lsrgan_x4` model weights `./results/pretrained_models/LSRGAN_x2-DIV2K-e19a5cef.pth.tar` successfully.
SR image save to `./figure/comic_sr.png`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### Optimizing Generative Adversarial Networks for Image Super Resolution via Latent Space Regularization

_Juncheng Li, Faming Fang, Kangfu Mei, Guixu Zhang_ <br>

**Abstract** <br>
Recent studies have shown that deep neural networks can significantly improve the quality of single-image
super-resolution. Current researches tend to use deeper convolutional neural networks to enhance performance. However,
blindly increasing the depth of the network cannot ameliorate the network effectively. Worse still, with the depth of
the network increases, more problems occurred in the training process and more training tricks are needed. In this
paper, we propose a novel multi-scale residual network (LSRGAN) to fully exploit the image features, which outperform most
of the state-of-the-art methods. Based on the residual block, we introduce convolution kernels of different sizes to
adaptively detect the image features in different scales. Meanwhile, we let these features interact with each other to
get the most efficacious image information, we call this structure Multi-scale Residual Block (MSRB). Furthermore, the
outputs of each MSRB are used as the hierarchical features for global feature fusion. Finally, all these features are
sent to the reconstruction module for recovering the high-quality image.

[[Paper]](https://arxiv.org/pdf/2001.08126.pdf)

```bibtex
@inproceedings{li2018multi,
  title={Multi-scale residual network for image super-resolution},
  author={Li, Juncheng and Fang, Faming and Mei, Kangfu and Zhang, Guixu},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={517--532},
  year={2018}
}
```
