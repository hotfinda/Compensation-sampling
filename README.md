# Compensation-sampling
This is the codebase for our paper [Compensation Sampling for Improved Convergence in Diffusion Models](https://arxiv.org/abs/2312.0628)

The repository is based on [DDIM](https://github.com/ermongroup/ddim) tuned by [ADM](https://github.com/openai/guided-diffusion) with our compensation sampling approach.

## Installation
We use the same installation as [ADM](https://github.com/openai/guided-diffusion)

```bash
git clone https://github.com/forever208/DDPM-IP.git
cd DDPM-IP
conda create -n ADM python=3.8
conda activate ADM
pip install -e .

# install the missing packages
conda install mpi4py
conda install numpy
pip install Pillow
pip install opencv-python

##Preparing Data and ADM base models.
The training code reads images from a directory of image files. We have prepared the codes in script folder to download datasets.

[ADM](https://github.com/openai/guided-diffusion)
