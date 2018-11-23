# ANIME-FACE-GAN

### Implement

#### SA-GAN

https://arxiv.org/pdf/1805.08318.pdf

Implemented with Attention, Conv2DTranspose, hinge loss and spectral norm.

The SAGAN was trained in batchsize=64 and cost only 3GB GPU memory. 


### Model Records

SAGAN_V2: SAGAN + deconv

SAGAN_V3: SAGAN + deconv + bs=64 + truncated_normal

SAGAN_V4: SAGAN + upsample + bs=128 + truncated_normal

SResNetGAN_V0: SResNetGAN + pixelshuffler

SResNetGAN_V1: SResNetGAN + deconv

### Experience

- Use truncated norm (std=0.5, truncated from -1 to 1) instead of uniform and Gaussian normal can help convergence.

- Binomial distribution works badly.

- Use Conv2DTranspose instead of Upsampling can improve the quality of images, and Upsampling also loses some diversities.

- It is normal that the hinge loss of discriminator equals 0 occasionally during the training.

- Bigger batch size (128, 256,...) dosen't achieve better performance in this project (not sure).

- Ensure enough steps to train (at least 60k).

