from layers import *
from tensor2tensor.layers.common_layers import upscale
import numpy as np


class SAGAN_model(object):
    def __init__(self, args):
        self.args = args
        self.layer_num = int(np.log2(self.args.img_size[0])) - 3

        # inputs
        self.is_training = tf.placeholder_with_default(False, (), name='is_training')
        self.inputs = tf.placeholder(tf.float32,
                                     [None, self.args.img_size[0], self.args.img_size[1], self.args.img_size[2]],
                                     name='inputs')
        self.z = tf.placeholder(tf.float32, [None, 1, 1, self.args.z_dim], name='z')  # noise

        # output of D for real images
        real_logits = self.discriminator(self.inputs)

        # output of D for fake images
        self.fake_images = self.generator(self.z)
        fake_logits = self.discriminator(self.fake_images, reuse=True)

        # get loss for discriminator
        self.d_loss = self.discriminator_loss(d_logits_real=real_logits, d_logits_fake=fake_logits)

        # get loss for generator
        self.g_loss = self.generator_loss(d_logits_fake=fake_logits)

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # optimizers
        self.d_opt = tf.train.AdamOptimizer(self.args.d_lr, beta1=self.args.beta1,
                                            beta2=self.args.beta2).minimize(self.d_loss, var_list=d_vars)
        self.g_opt = tf.train.AdamOptimizer(self.args.g_lr, beta1=self.args.beta1,
                                            beta2=self.args.beta2).minimize(self.g_loss, var_list=g_vars)

    def discriminator_loss(self, d_logits_real, d_logits_fake):
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - d_logits_real))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + d_logits_fake))
        loss = real_loss + fake_loss
        return loss

    def generator_loss(self, d_logits_fake):
        loss = -tf.reduce_mean(d_logits_fake)
        return loss

    def generator(self, z, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            ch = self.args.g_filters
            x = spectral_deconv2d(z, filters=ch, kernel_size=4, stride=1, is_training=self.is_training, padding='VALID',
                                  use_bias=False, scope='deconv2d')
            x = batch_norm(x, self.is_training, scope='batch_norm')
            x = tf.nn.relu(x)

            for i in range(self.layer_num // 2):
                if self.args.up_sample:
                    x = upscale(x, f=2)
                    x = spectral_conv2d(x, filters=ch // 2, kernel_size=3, stride=1, is_training=self.is_training,
                                        padding='SAME', scope='up_conv2d_' + str(i))
                else:
                    x = spectral_deconv2d(x, filters=ch // 2, kernel_size=4, stride=2, is_training=self.is_training,
                                          use_bias=False, scope='deconv2d_' + str(i))
                x = batch_norm(x, self.is_training, scope='batch_norm_' + str(i))
                x = tf.nn.relu(x)

                ch = ch // 2

            # Self Attention
            x = attention(x, ch, is_training=self.is_training, scope="attention", reuse=reuse)

            for i in range(self.layer_num // 2, self.layer_num):
                if self.args.up_sample:
                    x = upscale(x, f=2)
                    x = spectral_conv2d(x, filters=ch // 2, kernel_size=3, stride=1, is_training=self.is_training,
                                        padding='SAME', scope='up_conv2d_' + str(i))

                else:
                    x = spectral_deconv2d(x, filters=ch // 2, kernel_size=4, stride=2, is_training=self.is_training,
                                          use_bias=False, scope='deconv2d_' + str(i))
                x = batch_norm(x, self.is_training, scope='batch_norm_' + str(i))
                x = tf.nn.relu(x)

                ch = ch // 2

            if self.args.up_sample:
                x = upscale(x, f=2)
                x = spectral_conv2d(x, filters=self.args.img_size[2], kernel_size=3, stride=1, is_training=self.is_training,
                                    padding='SAME', scope='G_conv_logit')
            else:
                x = spectral_deconv2d(x, filters=self.args.img_size[2], kernel_size=4, stride=2, is_training=self.is_training,
                                      use_bias=False, scope='G_deconv_logit')
            x = tf.nn.tanh(x)

            return x

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = self.args.d_filters
            x = spectral_conv2d(x, filters=ch, kernel_size=4, stride=2, is_training=self.is_training, padding='SAME',
                                use_bias=False, scope='conv2d')
            x = tf.nn.leaky_relu(x, alpha=0.2)

            for i in range(self.layer_num // 2):
                x = spectral_conv2d(x, filters=ch * 2, kernel_size=4, stride=2, is_training=self.is_training,
                                    padding='SAME', use_bias=False,
                                    scope='conv2d_' + str(i))
                x = batch_norm(x, self.is_training, scope='batch_norm' + str(i))
                x = tf.nn.leaky_relu(x, alpha=0.2)

                ch = ch * 2

            # Self Attention
            x = attention(x, ch, is_training=self.is_training, scope="attention", reuse=reuse)

            for i in range(self.layer_num // 2, self.layer_num):
                x = spectral_conv2d(x, filters=ch * 2, kernel_size=4, stride=2, is_training=self.is_training,
                                    padding='SAME', use_bias=False,
                                    scope='conv2d_' + str(i))
                x = batch_norm(x, self.is_training, scope='batch_norm' + str(i))
                x = tf.nn.leaky_relu(x, alpha=0.2)

                ch = ch * 2

            x = spectral_conv2d(x, filters=1, kernel_size=4, padding='VALID', stride=1, is_training=self.is_training,
                                use_bias=False,
                                scope='D_logit')
            x = tf.squeeze(x, axis=[1, 2])

            return x
