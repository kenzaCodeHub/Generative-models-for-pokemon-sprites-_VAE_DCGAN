# Generative Models for Pokemon Sprites — VAE vs DCGAN

A comparative study of two generative architectures trained on the [Pokemon sprites dataset](https://pokemondb.net/sprites): a **Variational Autoencoder (VAE)** and a **Deep Convolutional GAN (DCGAN)**.

*Architecture and training setup adapted from [d2l.ai](https://d2l.ai) generative models chapter.*

## Objective

Explore the strengths and weaknesses of each generative architecture for sprite generation, and compare results across different hyperparameter configurations.

## Methods

### Variational Autoencoder (VAE)
- 4 conv layers in the encoder (3×64×64 → 256×4×4) + Gaussian bottleneck (μ, log σ²)
- 4 deconv layers in the decoder
- Loss: MSE reconstruction + KL divergence
- 3 configurations compared (latent_dim ∈ {64, 128, 256})

### Deep Convolutional GAN (DCGAN)
- Generator: 4 transposed convolutions with BatchNorm + ReLU, Tanh output
- Discriminator: mirror architecture with strided convolutions + LeakyReLU
- **Stabilization tricks:**
  - One-sided label smoothing (real labels = 0.9 instead of 1.0)
  - DCGAN-style weight initialization (normal, std=0.02)
  - Adam optimizer with `betas=(0.5, 0.999)`
- 2 configurations compared

## Dataset

Pokemon sprites — ~900 RGB images at 64×64 resolution, downloaded from d2l.ai.

## Results

### VAE — Final Training Loss

| Config | Latent dim | Epochs | Final Loss |
|---|---|---|---|
| A | 128 | 50 | 854.1 |
| B | 256 | 100 | **836.3** |
| C | 64 | 50 | 853.7 |

The largest latent dimension (256) trained for 100 epochs achieves the best reconstruction loss. All three configurations converge stably.

### DCGAN — Training Behavior

| Config | Latent dim | Epochs | Final loss_D | Final loss_G |
|---|---|---|---|---|
| A | 100 | 100 | 0.36 | 5.38 |
| B | 128 | 80 | 0.40 | 5.75 |

The discriminator loss drops while the generator loss steadily increases — a classic sign that **the discriminator dominates the generator**, which is expected behavior for GANs trained on small datasets (~900 images here).

## Key Findings

- **VAE** produces blurry but recognizable reconstructions (typical of MSE-trained VAEs)
- **DCGAN** generates sharper samples but suffers from discriminator dominance and mode collapse on this small dataset
- **Trade-off**: VAE = stable training + reconstructable latent space; GAN = sharp samples + unstable training

## Tech Stack

Python, PyTorch, torchvision, Matplotlib, Google Colab (GPU)

## Possible Improvements

- Perceptual loss (VGG features) instead of MSE for the VAE
- WGAN-GP or StyleGAN2 architecture for sharper GAN samples
- Heavy data augmentation (flips, color jitter, rotations) given the small dataset size
- Differential learning rates: lower LR for the discriminator to balance training
- Transfer learning from a model pre-trained on a similar domain

## Reference

[Dive into Deep Learning (d2l.ai)](https://d2l.ai) — Chapter 17: Generative Adversarial Networks
