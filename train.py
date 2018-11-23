import SResNetGAN
import SAGAN
import argparse
from utils import util
import os
from glob import glob
import tensorflow as tf
import math
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = '7'


def parse_args():
    desc = "Self-Attention GAN"
    parser = argparse.ArgumentParser(description=desc)
    # important!!!
    parser.add_argument('--name', type=str, default='SResNetGAN_V1', help='model name')

    parser.add_argument('--img_size', type=list, default=[128, 128, 3], help='img size')
    parser.add_argument('--fix_z', type=bool, default=False, help='whether fix the z feature')
    parser.add_argument('--std', type=float, default=0.5, help='learning rate for discriminator')
    parser.add_argument('--z_dim', type=int, default=128, help='dim of noises')

    # for SAGAN
    parser.add_argument('--d_filters', type=int, default=64, help='The filters in discriminator')
    parser.add_argument('--g_filters', type=int, default=1024, help='The filters in generator')
    parser.add_argument('--up_sample', type=bool, default=False, help='upsampling or deconv in generator')

    # for SResNetGAN
    parser.add_argument('--g_layer_num', type=int, default=16, help='layer num in generator')

    parser.add_argument('--epochs', type=int, default=400, help='Total epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--show_steps', type=int, default=100, help='steps to show imgs')
    parser.add_argument('--sample_num', type=int, default=64, help='num of imgs to show')
    parser.add_argument('--max_to_keep', type=int, default=10, help='num of saved models to keep')

    parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=4e-4, help='learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')

    parser.add_argument('--tfr_dir', type=str, default='dataset/tfrs/', help='dir to save tfrs')
    parser.add_argument('--input_dir', type=str, default='dataset/GetChu_aligned/',
                        help='the dir of input imgs with preprocess')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='log/', help='loss log')
    parser.add_argument('--restore', type=bool, default=False, help='whether to load trained models')
    parser.add_argument('--result_dir', type=str, default='results/',
                        help='Directory name to save the generated images')
    return check_args(parser.parse_args())


def check_args(args):
    args.checkpoint_dir += args.name
    args.result_dir += args.name
    assert args.sample_num % math.sqrt(args.sample_num) == 0
    assert os.path.exists(args.input_dir)
    util.check_folder(args.tfr_dir)
    util.check_folder(args.checkpoint_dir)
    util.check_folder(args.result_dir)
    util.check_folder(args.log_dir)
    return args


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    # parse arguments
    args = parse_args()

    # loading train data
    tf.logging.info('loading data...')
    img_paths = glob(args.input_dir + '*')
    total_num = len(img_paths)
    tfr_name = 'imgtfr_num' + str(total_num) + '_size' + str(args.img_size) + '.tf_record'
    if not os.path.exists(args.tfr_dir + tfr_name):
        tf.logging.warning('tfrecord is not found...generating...')
        util.get_tfrecords(img_paths=img_paths, output_dir=args.tfr_dir + tfr_name, img_size=args.img_size)
    train_dataset = util.input_fn(input_file=args.tfr_dir + tfr_name, batch_size=args.batch_size,
                                  img_size=args.img_size, buffer_size=20000)
    train_iterator = train_dataset.make_initializable_iterator()
    train_next_element = train_iterator.get_next()

    # open session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # model = SAGAN.SAGAN_model(args)
        model = SAGAN.SAGAN_model(args)
        tf.logging.info('model init over...')

        # show network architecture
        util.show_all_variables()

        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator.initializer)
        if args.restore:
            util.init_from_checkpoint(args.checkpoint_dir)
        # saver to save model
        saver = tf.train.Saver(max_to_keep=args.max_to_keep)

        n_batch = total_num // args.batch_size
        global_step = 0
        truncated_norm = util.get_truncated_normal(0.0, args.std)
        # fixed z
        z_fix = None
        if args.fix_z:
            z_fix = truncated_norm.rvs([args.batch_size, 1, 1, args.z_dim])
        for i_epoch in range(args.epochs):
            tf.logging.info("***** Epoch %d *****", i_epoch + 1)
            # training
            global_step, d_loss_log, g_loss_log = model.train_epoch(sess, train_next_element, i_epoch, n_batch,
                                                                    global_step, truncated_norm, z_fix)
            result = pd.DataFrame([d_loss_log, g_loss_log], index=['d_loss', 'g_loss']).transpose()
            result.to_csv(args.log_dir + args.name + '_results.csv', index=None)

            if i_epoch % 10 == 0:
                tf.logging.info('Saving model...')
                saver.save(sess, os.path.join(args.checkpoint_dir, 'model.ckpt'), global_step=global_step)


if __name__ == '__main__':
    main()
