# import tensorflow as tf
# from tensor2tensor.layers.common_layers import shape_list
# import numpy as np
#
# is_training = tf.placeholder_with_default(True, (), name='is_training')
# x = tf.Variable([1])
# x2 = tf.Variable([2])
# x3 = x
# with tf.control_dependencies([tf.cond(is_training, true_fn=lambda: x.assign(x2), false_fn=lambda: x.assign(x))]):
#     x3 = tf.reshape(x3, [1])
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(x3, feed_dict={is_training:True})) #

# test save_imgs
# from utils.util import save_images
# import numpy as np
#
# x = []
# flag = True
# for i in range(25):
#     if i % 2 == 0 and flag:
#         x.append((np.zeros((1, 128, 128, 3)) * 1.) / 2.)
#     else:
#         x.append(np.ones((1, 128, 128, 3)))
#
# X = np.concatenate(x, axis=0)
# save_images(X, [5, 5], image_path='test_save.jpg')

# test tfrecords
import tensorflow as tf
from utils import util
from glob import glob
import matplotlib.pyplot as plt

# img_paths = glob('dataset/temp_getchu/*')
# util.get_tfrecords(img_paths=img_paths, output_dir='test_tfr.tf_record', img_size=(128,128,3))
train_dataset = util.input_fn(input_file='test_tfr.tf_record', batch_size=32,
                                  img_size=(128,128,3), buffer_size=100)
train_iterator = train_dataset.make_initializable_iterator()
train_next_element = train_iterator.get_next()
with tf.Session() as sess:
    sess.run(train_iterator.initializer)
    batch_img = sess.run(train_next_element)
    print(batch_img.shape)
    img = batch_img[0]
    plt.imsave('test_pic.jpg', img)