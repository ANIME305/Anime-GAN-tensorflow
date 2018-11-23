# ANIME-FACE-GAN

### Implement

#### SA-GAN

https://arxiv.org/pdf/1805.08318.pdf

Implemented with Attention, Conv2DTranspose, hinge loss and spectral norm.

The SAGAN was trained in batchsize=64 and cost only 3GB GPU memory. 

#### results

61600 steps

![61600 steps](pictures/fake_steps_61600.jpg)

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

- Bigger batch size (128, 256,...) dosen't achieve better performance in this project (not sure).

- Ensure enough steps to train (at least 50k).

### Questions

- Pixelshuffle works bad (pool diversity).

- The hinge loss of discriminator usually equals 0 during the second half of training.

### TODO

- Add ExponentialMovingAverage to the generator.

- Add labels from illustration2vec.

- Learning rate exponentially decease after 50000 iterations of training.

