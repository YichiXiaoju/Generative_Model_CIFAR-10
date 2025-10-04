# Generative Models on CIFAR-10

This repository provides a modular implementation of several generative models, including:

- **DDPM** (Denoising Diffusion Probabilistic Models)  
- **DDIM** (Denoising Diffusion Implicit Models)  
- (Planned) **Flow Matching** and **Rectified Flow**

The goal is to train and compare different diffusion and flow-based generative models on CIFAR-10 in a unified framework.

## Project Structure

Generative_Model_CIFAR-10/
│
├── datasets/                # Data loading and preprocessing
│   └── cifar10.py
│
├── methods/                 # Training losses and samplers for each method
│   ├── base.py
│   ├── ddpm.py
│   └── ddim.py              # (new) DDIM sampler
│
├── models/                  # Model definitions
│   └── unet.py              # Compact UNet for CIFAR-10
│
├── utils/                   # Utility functions
│   ├── schedules.py         # Beta schedules (linear, cosine)
│   ├── embeddings.py
│   ├── ops.py
│   └── ema.py               # Exponential Moving Average
│
├── train.py                 # Training entry point
├── ddpm_samnple.py           # DDPM sampling script
├── ddim_sample.py           # DDIM sampling script
└── readme.md


