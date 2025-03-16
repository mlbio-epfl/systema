#!/bin/bash

# Create conda environment without confirmation
conda create -n scgpt python=3.10 -y

# Activate the environment
source activate scgpt

# Install PyTorch and dependencies without confirmation
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install CUDA from NVIDIA channel
conda install cuda -c nvidia/label/cuda-11.7.1 -y

# Install R base
conda install r-base=3.6.1 -y

# Install required Python packages via pip
pip install scgpt "flash-attn<1.0.5"
pip install packaging scvi-tools ninja scanpy leidenalg louvain learn2learn

# Uninstall and reinstall numba to ensure the latest compatible version
pip uninstall -y numba
pip install -U numba

pip install torch_geometric --no-deps
pip install psutil --no-deps
conda install ipykernel