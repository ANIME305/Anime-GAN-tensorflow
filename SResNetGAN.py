from layers import *
import numpy as np
import time
from utils import util
import os


class SResNetGAN_model(object):
    def __init__(self, args):
        self.args = args
        self.d_loss_log = []
        self.g_loss_log = []

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
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(update_ops):
            d_grads = tf.gradients(self.d_loss, d_vars)
            self.d_grads_norm = tf.zeros([1])
            for g in d_grads:
                self.d_grads_norm += tf.norm(g)
            d_opt = tf.train.AdamOptimizer(self.args.d_lr, beta1=self.args.beta1, beta2=self.args.beta2)
            self.train_d = d_opt.apply_gradients(zip(d_grads, d_vars), global_step=global_step)

            g_grads = tf.gradients(self.g_loss, g_vars)
            self.g_grads_norm = tf.zeros([1])
            for g in g_grads:
                self.g_grads_norm += tf.norm(g)
            g_opt = tf.train.AdamOptimizer(self.args.g_lr, beta1=self.args.beta1, beta2=self.args.beta2)
            self.train_g = g_opt.apply_gradients(zip(g_grads, g_vars), global_step=global_step)

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
            x = spectral_deconv2d(z, filters=64, kernel_size=16, stride=1, is_training=self.is_training,
                                  padding='VALID', use_bias=False, scope='deconv2d')
            x = batch_norm(x, self.is_training, scope='batch_norm')
            x = prelu(x)

            stage1_output = x

            for i in range(self.args.g_layer_num):
                x = residual_block(x, output_channel=64, stride=1, is_training=self.is_training,
                                   scope='residual_' + str(i + 1))

            # Self Attention
            x = attention(x, 64, is_training=self.is_training, scope="attention", reuse=reuse)

            with tf.variable_scope('resblock_output'):
                x = spectral_conv2d(x, 64, 3, 1, is_training=self.is_training, use_bias=False, scope='conv')
                x = batch_norm(x, self.is_training)

            x = x + stage1_output

            for i in range(int(np.log2(self.args.img_size[0] // 16))):
                with tf.variable_scope('subpixelconv_stage' + str(i + 1)):
                    x = spectral_conv2d(x, 256, 3, 1, is_training=self.is_training, scope='conv')
                    x = PixelShuffler(x, scale=2)
                    x = batch_norm(x, self.is_training)
                    x = prelu(x)

            x = spectral_conv2d(x, filters=self.args.img_size[2], kernel_size=9, stride=1, is_training=self.is_training,
                                padding='SAME', scope='G_conv_logit')
            x = tf.nn.tanh(x)

            return x

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            with tf.variable_scope('blocks1'):
                x = spectral_conv2d(x, filters=32, kernel_size=4, stride=2, is_training=self.is_training,
                                    scope='conv2d')  # 64*64
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = discriminator_block(x, 32, 3, 1, is_training=self.is_training, scope='d_residual_1')
                x = discriminator_block(x, 32, 3, 1, is_training=self.is_training, scope='d_residual_2')

            with tf.variable_scope('blocks2'):
                x = spectral_conv2d(x, filters=64, kernel_size=4, stride=2, is_training=self.is_training,
                                    scope='conv2d')  # 32*32
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = discriminator_block(x, 64, 3, 1, is_training=self.is_training, scope='d_residual_3')
                x = discriminator_block(x, 64, 3, 1, is_training=self.is_training, scope='d_residual_4')

            with tf.variable_scope('blocks3'):
                x = spectral_conv2d(x, filters=128, kernel_size=4, stride=2, is_training=self.is_training,
                                    scope='conv2d')  # 16*16
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = discriminator_block(x, 128, 3, 1, is_training=self.is_training, scope='d_residual_5')
                x = discriminator_block(x, 128, 3, 1, is_training=self.is_training, scope='d_residual_6')

            # Self Attention
            x = attention(x, 128, is_training=self.is_training, scope="attention", reuse=reuse)

            with tf.variable_scope('blocks4'):
                x = spectral_conv2d(x, filters=256, kernel_size=4, stride=2, is_training=self.is_training,
                                    scope='conv2d')  # 8*8
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = discriminator_block(x, 256, 3, 1, is_training=self.is_training, scope='d_residual_7')
                x = discriminator_block(x, 256, 3, 1, is_training=self.is_training, scope='d_residual_8')

            with tf.variable_scope('blocks5'):
                x = spectral_conv2d(x, filters=512, kernel_size=4, stride=2, is_training=self.is_training,
                                    scope='conv2d')  # 4*4
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = discriminator_block(x, 512, 3, 1, is_training=self.is_training, scope='d_residual_9')
                x = discriminator_block(x, 512, 3, 1, is_training=self.is_training, scope='d_residual_10')

            x = spectral_conv2d(x, filters=1, kernel_size=4, padding='VALID', stride=1, is_training=self.is_training,
                                use_bias=False,
                                scope='D_logit')
            x = tf.squeeze(x, axis=[1, 2])

            return x

    def preprocess(self, x):
        x = x / 127.5 - 1
        return x

    def train_epoch(self, sess, train_next_element, i_epoch, n_batch, global_step, truncated_norm, z_fix=None):
        t_start = None
        for i_batch in range(n_batch):
            if i_batch == 1:
                t_start = time.time()
            batch_imgs = sess.run(train_next_element)
            batch_imgs = self.preprocess(batch_imgs)
            batch_z = truncated_norm.rvs([self.args.batch_size, 1, 1, self.args.z_dim])
            feed_dict_ = {self.inputs: batch_imgs,
                          self.z: batch_z,
                          self.is_training: True}
            # update D network
            _, d_loss, d_grads_norm = sess.run([self.train_d, self.d_loss, self.d_grads_norm], feed_dict=feed_dict_)
            self.d_loss_log.append(d_loss)

            # update G network
            g_loss = None
            g_grads_norm = None
            if i_batch % self.args.n_critic == 0:
                _, g_loss, g_grads_norm = sess.run([self.train_g, self.g_loss, self.g_grads_norm], feed_dict=feed_dict_)
                self.g_loss_log.append(g_loss)

            global_step += 1

            last_train_str = "[epoch:%d/%d, global_step:%d] -d_loss:%.3f - g_loss:%.3f -d_norm:%.3f -g_norm:%.3f" % (
                i_epoch + 1, int(self.args.epochs), global_step, d_loss, g_loss, d_grads_norm, g_grads_norm)
            if i_batch > 0:
                last_train_str += (' -ETA:%ds' % util.cal_ETA(t_start, i_batch, n_batch))
            if (i_batch + 1) % 10 == 0 or i_batch == 0:
                tf.logging.info(last_train_str)

            # show fake_imgs
            if global_step % self.args.show_steps == 0:
                tf.logging.info('generating fake imgs in steps %d...' % global_step)

                if z_fix is not None:
                    show_z = z_fix
                else:
                    show_z = batch_z
                fake_imgs = sess.run(self.fake_images, feed_dict={self.z: show_z})
                manifold_h = int(np.floor(np.sqrt(self.args.sample_num)))
                util.save_images(fake_imgs, [manifold_h, manifold_h],
                                 image_path=os.path.join(self.args.result_dir,
                                                         'fake_steps_' + str(global_step) + '.jpg'))

        return global_step, self.d_loss_log, self.g_loss_log
