import argparse
import os


def str2bool(x):
    return x.lower() in 'true'


parse = argparse.ArgumentParser()
parse.add_argument('--phase', default='train', help='train/test')
parse.add_argument('--log_dir', default='log')
parse.add_argument('--data_dir', default='data')
parse.add_argument('--augment', type=str2bool, default=True)
parse.add_argument('--restore', action='store_true')
parse.add_argument('--more_up', type=int, default=2)

parse.add_argument('--training_epoch', type=int, default=101)

parse.add_argument('--batch_size', type=int, default=28)

parse.add_argument('--use_non_uniform', type=str2bool, default=True)
parse.add_argument('--jitter', type=str2bool, default=False)
parse.add_argument('--jitter_sigma', type=float, default=0.01, help='jitter augmentation')
parse.add_argument('--jitter_max', type=float, default=0.03, help='jitter augmentation')

parse.add_argument('--up_ratio', type=int, default=4)

parse.add_argument('--num_point', type=int, default=256)
parse.add_argument('--patch_num_point', type=int, default=256)
parse.add_argument('--patch_num_ratio', type=int, default=3)

parse.add_argument('--base_lr_d', type=float, default=0.0001)
parse.add_argument('--base_lr_g', type=float, default=0.001)

parse.add_argument('--beta', type=float, default=0.9)

parse.add_argument('--start_decay_step', type=int, default=50000)

parse.add_argument('--lr_decay_steps', type=int, default=50000)
parse.add_argument('--lr_decay_rate', type=float, default=0.7)
parse.add_argument('--lr_clip', type=float, default=1e-6)

parse.add_argument('--steps_per_print', type=int, default=1)

parse.add_argument('--visualize', type=str2bool, default=False)
parse.add_argument('--steps_per_visu', type=int, default=100)

parse.add_argument('--epoch_per_save', type=int, default=5)

parse.add_argument('--use_repulse', type=str2bool, default=True)
parse.add_argument('--repulsion_w', default=1.0, type=float, help="repulsion_weight")

parse.add_argument('--fidelity_w', default=100.0, type=float, help="fidelity_weight")
# 均匀损失权重
parse.add_argument('--uniform_w', default=10.0, type=float, help="uniform_weight")
# gan损失权重
parse.add_argument('--gan_w', default=0.5, type=float, help="gan_weight")
parse.add_argument('--gen_update', default=2, type=int, help="gen_update")

FLAGS = parse.parse_args()