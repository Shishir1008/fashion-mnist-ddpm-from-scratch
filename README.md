# Fashion-MNIST DDPM From Scratch

A small PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** built from scratch and trained on a subset of Fashion-MNIST.

The goal of this project was not to build a production-quality image generator. Instead, I wanted to understand what is actually happening inside a diffusion model — from adding noise to an image, to training a U-Net to predict that noise, and finally using the learned model to denoise an image.

The experiment uses a deliberately small dataset so that the complete diffusion pipeline can be inspected and understood in a single notebook.

## What this project demonstrates

The notebook implements the main components of a DDPM without relying on a pre-built diffusion pipeline:

* Forward diffusion
* Linear beta noise schedule
* Cumulative alpha products
* Timestep-dependent noise addition
* Sinusoidal timestep embeddings
* Residual convolutional blocks
* Timestep-conditioned U-Net
* Skip connections
* Noise-prediction training objective
* DDPM epsilon prediction
* Deterministic DDIM-style reconstruction
* Reconstruction metrics
* Training-loss visualization
* Model checkpoint saving and loading

## The experiment

I trained the model using **200 Fashion-MNIST images** resized to `32×32`.

The model is unconditional: although Fashion-MNIST provides class labels, the labels are not given to the diffusion model during training.

The main experiment is controlled reconstruction rather than generation from pure Gaussian noise.

The process is:

```text
Original image
      │
      ▼
Forward diffusion
      │
      ▼
Noisy image xₜ
      │
      │  timestep t
      ▼
   U-Net
      │
      ▼
Predicted noise εθ(xₜ,t)
      │
      ▼
Reverse denoising
      │
      ▼
Reconstructed image
```

This setup makes it easier to observe whether the model has learned to remove noise while preserving the main structure of the original image.

## Model architecture

The denoising network is a compact timestep-conditioned U-Net designed for `32×32` grayscale images.

The spatial resolution follows:

```text
32×32
  ↓
16×16
  ↓
 8×8
  ↓
16×16
  ↓
32×32
```

The network contains:

* Convolutional input/output layers
* Residual blocks
* Group Normalization
* SiLU activations
* Sinusoidal timestep embeddings
* Timestep projections inside residual blocks
* Downsampling and upsampling layers
* U-Net skip connections

The timestep is important because the amount of noise changes throughout the diffusion process. The model therefore receives an embedding of the current timestep and uses it inside its residual blocks.

## Training objective

The model is trained using the standard epsilon-prediction objective.

For an original image `x₀`, a timestep `t`, and Gaussian noise `ε`, the forward process produces:

```text
xₜ = √α̅ₜ x₀ + √(1 - α̅ₜ) ε
```

The U-Net receives `xₜ` and `t` and predicts the noise:

```text
εθ(xₜ, t)
```

The training objective is the mean squared error between the actual noise and the predicted noise:

```text
L = MSE(ε, εθ(xₜ, t))
```

During training, a random timestep is sampled for each image in a batch, so the model learns to predict noise at different stages of the diffusion process.

## Configuration

The main experiment uses:

| Setting                       |         Value |
| ----------------------------- | ------------: |
| Dataset                       | Fashion-MNIST |
| Training images               |           200 |
| Image size                    |         32×32 |
| Channels                      |             1 |
| Batch size                    |            32 |
| Diffusion timesteps           |         1,000 |
| Beta schedule                 |        Linear |
| Beta start                    |        0.0001 |
| Beta end                      |          0.02 |
| Base U-Net channels           |            32 |
| Learning rate                 |        2×10⁻⁴ |
| Epochs                        |           200 |
| Reconstruction timestep       |           150 |
| Reconstruction sampling steps |            50 |
| Optimizer                     |          Adam |
| Framework                     |       PyTorch |

The notebook was run using a CUDA-enabled environment; the recorded run used a Tesla T4 GPU.

## Forward diffusion

One of the first experiments visualizes how the same Fashion-MNIST image changes as the timestep increases.

```text
t = 0       → almost original image
t = 50      → slightly noisy
t = 150     → visibly corrupted
t = 300     → heavily corrupted
t = 600     → mostly noise
t = 999     → approximately pure noise
```

This makes the forward diffusion process easier to understand visually rather than only through the mathematical formulation.

## Reconstruction

For reconstruction, I start with a real Fashion-MNIST image and add noise at a controlled timestep.

The trained model then performs deterministic reverse denoising.

```text
Original
   │
   ▼
Add Gaussian noise
   │
   ▼
Noisy image at t = 150
   │
   ▼
U-Net predicts noise
   │
   ▼
Deterministic reverse process
   │
   ▼
Denoised reconstruction
```

The reconstruction uses a subset of the original 1,000-step training schedule, with 50 reverse sampling steps.

This is intentionally different from unconditional image generation.

The model is **not** being asked to start from pure Gaussian noise and invent a new Fashion-MNIST image. Instead, the experiment tests whether the learned denoising function can recover a moderately corrupted real image.

## Why use only 200 images?

This is intentionally a small experiment.

Using a small subset keeps the project computationally manageable while making the internal mechanics of the diffusion process easier to inspect.

It also means that the results should not be interpreted as a benchmark for DDPM performance. The purpose is primarily to understand and implement the underlying mechanism.

With a larger dataset and longer training, the same architecture could be developed into a more capable generative experiment.

## Results

The notebook produces several visualizations and measurements, including:

* Training images
* Forward diffusion at different timesteps
* Training loss
* Original → noisy → reconstructed comparisons
* Reconstruction results across multiple images
* MAE
* MSE
* PSNR

The notebook also saves the trained model checkpoint and the training configuration so the experiment can be reproduced or resumed.

## Repository structure

```text
fashion-mnist-ddpm-from-scratch/
│
├── fashion_mnist_ddpm_denoising_demo.ipynb
└── README.md
```

The implementation is intentionally kept in one notebook because this project is primarily an educational implementation and experiment. The notebook contains the complete pipeline from dataset preparation through training, reconstruction, visualization, and checkpointing.

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/Shishir1008/fashion-mnist-ddpm-from-scratch.git
cd fashion-mnist-ddpm-from-scratch
```

### 2. Install dependencies

```bash
pip install torch torchvision numpy matplotlib tqdm
```

### 3. Open the notebook

The easiest way to run the project is with Jupyter or Google Colab.

The notebook automatically downloads Fashion-MNIST through `torchvision` on the first run.

A CUDA-enabled GPU is recommended because training the model on CPU will be considerably slower.

## What I learned

This project helped me connect the mathematical description of diffusion models with an actual implementation.

In particular, I focused on understanding:

1. How the forward process gradually destroys image information.
2. Why the model predicts noise instead of directly predicting the clean image.
3. How the timestep is represented using sinusoidal embeddings.
4. How timestep information is injected into convolutional residual blocks.
5. Why U-Net skip connections are useful for preserving spatial information.
6. How the predicted noise can be used to estimate the clean image.
7. The difference between controlled reconstruction and unconditional generation.
8. How a 1,000-step training schedule can be sampled using fewer reverse steps during reconstruction.

## Limitations

This is a deliberately small implementation and has several limitations:

* Only 200 training images are used.
* The images are only `32×32`.
* The model is relatively small.
* Training is not intended to produce state-of-the-art samples.
* The main experiment focuses on reconstruction rather than unconditional generation.
* Reconstruction quality depends strongly on the amount of noise and the amount of training.

These limitations are intentional because the main purpose of the project is to understand the mechanics of DDPMs rather than optimize generation quality.

## References

The implementation is based on the core ideas introduced in:

**Ho, J., Jain, A., & Abbeel, P. (2020).
Denoising Diffusion Probabilistic Models.**

https://arxiv.org/abs/2006.11239

For the deterministic reconstruction procedure, the notebook uses the DDIM-style idea introduced in:

**Song, J., Meng, C., & Ermon, S. (2020).
Denoising Diffusion Implicit Models.**

https://arxiv.org/abs/2010.02502

---

### Notes

This repository is primarily a learning and implementation project. The emphasis is on understanding the internal mechanics of diffusion models rather than reproducing the scale or performance of modern image-generation systems.
