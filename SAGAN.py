from layers import *
import numpy as np
import time
from utils import util
import os


class SAGAN_model(object):
    def __init__(self, args):
        self.args = args
        self.d_loss_log = []
        self.g_loss_log = []
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

        # global step
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.add_step = self.global_step.assign(self.global_step + 1)

        # optimizers
        self.d_lr = tf.train.exponential_decay(self.args.d_lr,
                                               tf.maximum(self.global_step - self.args.decay_start_steps, 0),
                                               self.args.decay_steps,
                                               self.args.decay_rate)
        self.g_lr = tf.train.exponential_decay(self.args.g_lr,
                                               tf.maximum(self.global_step - self.args.decay_start_steps, 0),
                                               self.args.decay_steps,
                                               self.args.decay_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_grads = tf.gradients(self.d_loss, d_vars)
            d_opt = tf.train.AdamOptimizer(self.d_lr, beta1=self.args.beta1, beta2=self.args.beta2)
            self.train_d = d_opt.apply_gradients(zip(d_grads, d_vars))

            g_grads = tf.gradients(self.g_loss, g_vars)
            g_opt = tf.train.AdamOptimizer(self.g_lr, beta1=self.args.beta1, beta2=self.args.beta2)
            self.train_g = g_opt.apply_gradients(zip(g_grads, g_vars))

            # EMA for generator
            with tf.variable_scope("EMA_Weights"):
                if self.args.ema_decay is not None:
                    self.var_ema = tf.train.ExponentialMovingAverage(self.args.ema_decay, num_updates=self.global_step)
                    with tf.control_dependencies([self.train_g]):
                        self.ema_train_g = self.var_ema.apply(tf.trainable_variables(scope='generator'))
                    # assign ema weights
                    self.assign_vars = []
                    for var in tf.trainable_variables(scope='generator'):
                        v = self.var_ema.average(var)
                        if v is not None:
                            self.assign_vars.append(tf.assign(var, v))

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
            x = tf.nn.leaky_relu(x, alpha=0.2)

            for i in range(self.layer_num // 2):
                with tf.variable_scope('layer' + str(i)):
                    if self.args.up_sample:
                        x = up_sample(x, scale_factor=2)
                        x = spectral_conv2d(x, filters=ch // 2, kernel_size=3, stride=1, is_training=self.is_training,
                                            padding='SAME', scope='up_conv2d_' + str(i))
                    else:
                        x = spectral_deconv2d(x, filters=ch // 2, kernel_size=4, stride=2, is_training=self.is_training,
                                              use_bias=False, scope='deconv2d_' + str(i))
                    x = batch_norm(x, self.is_training, scope='batch_norm_' + str(i))
                    x = tf.nn.leaky_relu(x, alpha=0.2)

                    ch = ch // 2

            # Self Attention
            x = attention(x, ch, is_training=self.is_training, scope="attention", reuse=reuse)

            for i in range(self.layer_num // 2, self.layer_num):
                with tf.variable_scope('layer' + str(i)):
                    if self.args.up_sample:
                        x = up_sample(x, scale_factor=2)
                        x = spectral_conv2d(x, filters=ch // 2, kernel_size=3, stride=1, is_training=self.is_training,
                                            padding='SAME', scope='up_conv2d_' + str(i))
                    else:
                        x = spectral_deconv2d(x, filters=ch // 2, kernel_size=4, stride=2, is_training=self.is_training,
                                              use_bias=False, scope='deconv2d_' + str(i))
                    x = batch_norm(x, self.is_training, scope='batch_norm_' + str(i))
                    x = tf.nn.leaky_relu(x, alpha=0.2)

                    ch = ch // 2

            if self.args.up_sample:
                x = up_sample(x, scale_factor=2)
                x = spectral_conv2d(x, filters=self.args.img_size[2], kernel_size=3, stride=1,
                                    is_training=self.is_training,
                                    padding='SAME', scope='G_conv_logit')
            else:
                x = spectral_deconv2d(x, filters=self.args.img_size[2], kernel_size=4, stride=2,
                                      is_training=self.is_training,
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

    def preprocess(self, x):
        x = x / 127.5 - 1
        return x

    def train_epoch(self, sess, saver, train_next_element, i_epoch, n_batch, truncated_norm, z_fix=None):
        t_start = None
        global_step = 0
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
            _, d_loss, d_lr, g_lr = sess.run([self.train_d, self.d_loss, self.d_lr, self.g_lr], feed_dict=feed_dict_)
            self.d_loss_log.append(d_loss)

            # update G network
            g_loss = None
            if i_batch % self.args.n_critic == 0:
                if self.args.ema_decay is not None:
                    _, g_loss, _, global_step = sess.run(
                        [self.ema_train_g, self.g_loss, self.add_step, self.global_step], feed_dict=feed_dict_)
                else:
                    _, g_loss, _, global_step = sess.run([self.train_g, self.g_loss, self.add_step, self.global_step],
                                                         feed_dict=feed_dict_)
            self.g_loss_log.append(g_loss)

            last_train_str = "[epoch:%d/%d, global_step:%d] -d_loss:%.3f - g_loss:%.3f -d_lr:%.e -g_lr:%.e" % (
                i_epoch + 1, int(self.args.epochs), global_step, d_loss, g_loss, d_lr, g_lr)
            if i_batch > 0:
                last_train_str += (' -ETA:%ds' % util.cal_ETA(t_start, i_batch, n_batch))
            if (i_batch + 1) % 20 == 0 or i_batch == 0:
                tf.logging.info(last_train_str)

            # show fake_imgs
            if global_step % self.args.show_steps == 0:
                tf.logging.info('generating fake imgs in steps %d...' % global_step)
                # do ema
                if self.args.ema_decay is not None:
                    # save temp weights for generator
                    saver.save(sess, os.path.join(self.args.checkpoint_dir, 'temp_model.ckpt'))
                    sess.run(self.assign_vars, feed_dict={self.inputs: batch_imgs,
                                                          self.z:batch_z,
                                                          self.is_training: False})
                    tf.logging.info('After EMA...')

                if z_fix is not None:
                    show_z = z_fix
                else:
                    show_z = truncated_norm.rvs([self.args.batch_size, 1, 1, self.args.z_dim])
                fake_imgs = sess.run(self.fake_images, feed_dict={self.z: show_z})
                manifold_h = int(np.floor(np.sqrt(self.args.sample_num)))
                util.save_images(fake_imgs, [manifold_h, manifold_h],
                                 image_path=os.path.join(self.args.result_dir,
                                                         'fake_steps_' + str(global_step) + '.jpg'))
                if self.args.ema_decay is not None:
                    # restore temp weights for generator
                    saver.restore(sess, os.path.join(self.args.checkpoint_dir, 'temp_model.ckpt'))
                    tf.logging.info('Recover weights over...')

        return global_step, self.d_loss_log, self.g_loss_log
