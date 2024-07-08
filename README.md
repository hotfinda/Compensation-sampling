# Compensation-sampling
This is the codebase for our paper [Compensation Sampling for Improved Convergence in Diffusion Models](https://arxiv.org/abs/2312.06285)

The repository is based on [DDIM](https://github.com/ermongroup/ddim) tuned by [ADM](https://github.com/openai/guided-diffusion) with our compensation sampling approach.

## Installation
We use the same installation as [ADM](https://github.com/openai/guided-diffusion)

```bash
git clone https://github.com/forever208/DDIM-IP.git
cd DDIM-IP
conda create -n ADM python=3.8
conda activate ADM
pip install -e .

# install the missing packages
conda install mpi4py
conda install numpy
pip install Pillow
pip install opencv-python
```

## Preparing Data and ADM base models.
The training code reads images from a directory of image files. We have prepared the codes in script folder to download datasets.
For using models during training, please download the corresponding [model card](https://github.com/openai/guided-diffusion).

## Training models

The scripts are based on [cold diffusion](https://github.com/arpitbansal297/Cold-Diffusion-Models), and we have separate scripts for training models on each dataset i.e <dataset>_<resolution>.py. 

The --time_steps argument can used to vary the number of steps it takes to reach the final isotropic Gaussian noise distribution.
The --sampling_routine argument allows you to switch between different sampling algorithms. Choosing default will sample using DDIM sampling, cold is sampling algorithm from Cold diffusion paper, and CS is our approach.

The --save_folder argument indicates the path to save the trained model, and the training data samples produced to keep track of progress. The frequency of saving and progress tracking can be modified in the Trainer class defined in denoising_diffusion_pytorch.py. The data_path argument specifies the path to the training data folder produced in the dataset preparation step.

Below is an example script for training denoising diffusion models.

```bash
python <dataset>_<resolution>.py --time_steps 200 --sampling_routine 'CS' --save_folder <Path to save model folder> --data_path <Path to train data folder>
```

## Testing models
Below is an example of generating iamges for testing denoising diffusion models:

```bash
python cifar10_test.py --time_steps 50 --sampling_routine 'CS' --save_folder <Path to save results> --data_path <Path to data folder> --test_type test_data
```

For testing the FID score, here is an example:
```bash
python cifar10_test.py --time_steps 50 --sampling_routine 'CS' --save_folder <Path to save results> --data_path <Path to data folder> --test_type 'test_sample_and_save_for_fid'
```



