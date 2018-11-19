import SAGAN
import argparse
from utils import util
import tensorflow as tf


def parse_args():
    desc = "Self-Attention GAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--img_size', type=list, default=[128, 128, 3], help='img size')
    parser.add_argument('--up_sample', type=bool, default=True, help='upsampling or deconv in generator')
    parser.add_argument('--z_dim', type=int, default=128, help='dim of noises')
    parser.add_argument('--d_filters', type=int, default=64, help='The filters in discriminator')
    parser.add_argument('--g_filters', type=int, default=1024, help='The filters in generator')

    parser.add_argument('--iterations', type=int, default=100000, help='Total training steps')
    parser.add_argument('--show_loss', type=int, default=100, help='steps show loss')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--show_steps', type=int, default=500, help='steps to show imgs')
    parser.add_argument('--sample_num', type=int, default=36, help='num of imgs to show')
    parser.add_argument('--save_steps', type=int, default=1000, help='steps to save the model')

    parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=4e-4, help='learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    return check_args(parser.parse_args())


def check_args(args):
    util.check_folder(args.checkpoint_dir)
    util.check_folder(args.result_dir)
    return args

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    # parse arguments
    args = parse_args()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = SAGAN.SAGAN_model(args)
        tf.logging.info('model init over...')

        # show network architecture
        util.show_all_variables()


if __name__ == '__main__':
    main()
