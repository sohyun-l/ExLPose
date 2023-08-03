# Human Pose Estimation in Extremely Low-light Conditions

### [Project Page](http://cg.postech.ac.kr/research/ExLPose/) | [Paper](https://arxiv.org/abs/2303.15410)
This repo is the official implementation of [**CVPR 2023**] paper: "[Human Pose Estimation in Extremely Low-light Conditions](https://arxiv.org/abs/2303.15410)".

> [Human Pose Estimation in Extremely Low-light Conditions]([https://arxiv.org/abs/2204.01587](https://arxiv.org/abs/2303.15410))     
> [Sohyun Lee](https://sohyun-l.github.io)<sup>1*</sup>, Jaesung Rim<sup>1*</sup>, Boseung Jeong<sup>1</sup>, Geonu Kim<sup>1</sup>, Byungju Woo<sup>2</sup>, Haechan Lee<sup>1</sup>, [Sunghyun Cho](https://www.scho.pe.kr/)<sup>1</sup>, [Suha Kwak](http://cvlab.postech.ac.kr/~suhakwak/)<sup>1</sup>\
> POSTECH<sup>1</sup> ADD<sup>2</sup>\
> CVPR 2023


## Overview
We study human pose estimation in extremely low-light images. This task is challenging due to the difficulty of collecting real low-light images with accurate labels, and severely corrupted inputs that degrade prediction quality significantly. To address the first issue, we develop a dedicated camera system and build a new dataset of real lowlight images with accurate pose labels. Thanks to our camera system, each low-light image in our dataset is coupled with an aligned well-lit image, which enables accurate pose labeling and is used as privileged information during training. We also propose a new model and a new training strategy that fully exploit the privileged information to learn representation insensitive to lighting conditions. Our method demonstrates outstanding performance on real extremely low-light images, and extensive analyses validate that both of our model and dataset contribute to the success.

## Citation
If you find our code or paper useful, please consider citing our paper:

```BibTeX
@inproceedings{lee2023human,
  title={Human pose estimation in extremely low-light conditions},
  author={Lee, Sohyun and Rim, Jaesung and Jeong, Boseung and Kim, Geonu and Woo, Byungju and Lee, Haechan and Cho, Sunghyun and Kwak, Suha},
  booktitle={Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

## Dataset
[Our Project Page](http://cg.postech.ac.kr/research/ExLPose/)

## Installation
This repository is developed and tested on

- Ubuntu 20.04
- Conda 4.9.2
- CUDA 11.4
- Python 3.7.11
- PyTorch 1.9.0

## Environment Setup
* Required environment is presented in the 'exlpose.yaml' file
* Clone this repo
```bash
~$ git clone https://github.com/sohyun-l/ExLPose
~$ cd ExLPose
~/ExLPose$ conda env create --file exlpose.yaml
~/ExLPose$ conda activate exlpose.yaml
```

