# Generative Model Comparison on CIFAR-10  
**DDPM | DDIM | Flow Matching | Rectified Flow**

This repository provides a unified PyTorch implementation of four modern generative model paradigms — **Diffusion Models (DDPM, DDIM)** and **Flow-based Models (Flow Matching, Rectified Flow)** — trained and compared on the **CIFAR-10** dataset.  
The codebase is modular, extensible, and numerically stable, featuring cosine noise schedules, EMA stabilization, and dynamic thresholding.

---

## Features

| Category | Description |
|-----------|-------------|
| **Dataset** | CIFAR-10, automatically downloaded via `torchvision.datasets` |
| **Model Backbone** | Full U-Net with time embedding and residual attention blocks |
| **Schedulers** | Linear and cosine β-schedules |
| **Sampling Methods** | DDPM (stochastic), DDIM (deterministic), Rectified Flow (ODE-based) |
| **Stabilization** | EMA (Exponential Moving Average), gradient clipping, dynamic thresholding |
| **Hardware Tested** | NVIDIA A100, RTX 4090, Colab T4 (FP32) |

---

## Model Overview

### 1. **DDPM** — *Denoising Diffusion Probabilistic Model*
- Learns to predict additive Gaussian noise ε from noisy images.
- Sampling follows posterior transition:
  \[
  p_\theta(x_{t-1}|x_t) = \mathcal{N}\big(\mu_\theta(x_t, t), \tilde{\beta}_t I \big)
  \]
- Implemented with **x₀-form mean** and **dynamic thresholding** for stable generation.

### 2. **DDIM** — *Deterministic Diffusion Implicit Model*
- Deterministic sampling derived from DDPM marginal consistency.
- ODE formulation enables fewer steps (e.g. 50–100) with similar quality.
- Excellent speed-quality tradeoff.

### 3. **Flow Matching (FM)**
- Trains a neural vector field \(v_\theta(x, t)\) to match the true probability flow ODE.
- Provides a direct continuous-time perspective bridging diffusion and flow models.

### 4. **Rectified Flow (RF)**
- A simplified flow-matching variant using a “rectified” linear trajectory.
- Sampling via ODE solver (Euler / Heun):
  \[
  x_{t+\Delta t} = x_t + \Delta t \, v_\theta(x_t, t)
  \]
- Deterministic, fast, and interpretable.

---

## 🧩 Repository Structure
```text
Generative_Model_CIFAR-10/
├── datasets/
│ └── cifar10.py # CIFAR-10 dataloader
├── models/
│ └── unet.py # Full U-Net backbone
├── methods/
│ ├── ddpm.py # Training loss for DDPM
│ ├── ddim.py # Sampling and ODE formulation for DDIM
│ ├── flow_matching.py # Flow Matching implementation
│ └── rectified_flow.py # Rectified Flow training and ODE solvers
├── utils/
│ ├── schedules.py # Linear / cosine beta schedules
│ ├── ema.py # Exponential Moving Average class
│ └── visualization.py # Optional visualization utilities
├── train.py # Unified training pipeline
├── ddpm_sample.py # Improved DDPM sampling (x0-form)
├── ddim_sample.py # Deterministic DDIM sampling
├── sample_rectifiedflow.py # Rectified Flow sampling
└── samples/
├── ddpm_16.png
├── ddim_16.png
└── rf_100.png



