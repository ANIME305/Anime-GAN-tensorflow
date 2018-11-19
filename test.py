import tensorflow as tf
from tensor2tensor.layers.common_layers import shape_list
import numpy as np

is_training = tf.placeholder_with_default(True, (), name='is_training')
x = tf.Variable([1])
x2 = tf.Variable([2])
x3 = x
with tf.control_dependencies([tf.cond(is_training, true_fn=lambda: x.assign(x2), false_fn=lambda: x.assign(x))]):
    x3 = tf.reshape(x3, [1])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(x3, feed_dict={is_training:True})) #
