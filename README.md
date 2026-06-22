# Latent Block-Diffusion Temporal Point Processes
Pytorch implementation of the paper "Latent Block-Diffusion Temporal Point Processes: A Semi-Autoregressive Framework for Asynchronous Event Sequence Generation".

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
