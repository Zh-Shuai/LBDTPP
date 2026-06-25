# Latent Block-Diffusion Temporal Point Processes

Pytorch implementation of the paper **"Latent Block-Diffusion Temporal Point Processes: A Semi-Autoregressive Framework for Asynchronous Event Sequence Generation"**.

> Paper: [arXiv:2606.24982](https://arxiv.org/abs/2606.24982)  

## Overview

LBDTPP is a semi-autoregressive framework for asynchronous event sequence generation. It models event sequences at two levels:

- **Across blocks:** event blocks are generated autoregressively, preserving temporal dependencies and supporting variable-length generation.
- **Within each block:** multiple event representations are sampled in parallel through Gaussian diffusion in a continuous latent space, reducing event-by-event error accumulation.

This design combines the length flexibility of autoregressive TPPs with the high-quality parallel generation capability of non-autoregressive diffusion models.

## Installation
1. Install the dependencies
```
conda create --name bdtpp python=3.10
conda activate bdtpp
pip install -r requirements.txt
```
2. Unzip the data
```
unzip data.zip
```

## Dataset
The six real-world datasets are from [CDiff](https://github.com/networkslab/cdiff).

## Running Experiments
We provide runnable experiment scripts:
- `exp_conditional.sh`: commands for conditional generation
- `exp_unconditional.sh`: commands for unconditional generation

## Citation

If you find this repository useful, please cite:

```bibtex
@article{zhang2026latent,
  title={Latent Block-Diffusion Temporal Point Processes: A Semi-Autoregressive Framework for Asynchronous Event Sequence Generation},
  author={Zhang, Shuai and Chen, Yancheng and Zhou, Chuan and Liu, Yang and Lin, Xixun and Zhao, Xiangyu and Zhu, Jun and Ma, Zhi-Ming},
  journal={arXiv preprint arXiv:2606.24982},
  year={2026}
}
```
