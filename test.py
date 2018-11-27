# TEST
# x=0.995
# for i in range(100):
#     x*=0.995
# print(x)

import tensorflow as tf
from layers import *
from utils import util

z = tf.placeholder(tf.float32, [None, 1, 1, 128], name='z')
is_training = tf.placeholder_with_default(False, (), name='is_training')
x = spectral_deconv2d(z, filters=1024, kernel_size=4, stride=1, is_training=is_training, padding='VALID',
                                  use_bias=False, scope='deconv2d')

util.show_all_variables()
print(x)
