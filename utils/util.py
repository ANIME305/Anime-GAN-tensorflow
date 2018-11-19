import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)