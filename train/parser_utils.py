from argparse import ArgumentParser, ArgumentTypeError
from distutils.util import strtobool

from consts import DEF_DATA_PATH


def subj_sess_tasks_triple(input_str):
    # taken from https://stackoverflow.com/questions/9978880/python-argument-parser-list-of-list-or-tuple-of-tuples
    try:
        x, y, z = input_str.split(',')
        return x, y, z
    except ValueError:
        raise ArgumentTypeError("Coordinates must be x,y,z")


def get_base_parser():
    parser = ArgumentParser()

    # dataset related
    parser.add_argument("--data_path", type=str, default=DEF_DATA_PATH)
    parser.add_argument("--dataset", type=str, default='schoffelen',
                        choices=['armeni', 'gwilliams', 'schoffelen', 'schar_combined'])

    # data selection/processing
    parser.add_argument("--cache_data", type=strtobool, nargs='?', const=True, default=True)
    parser.add_argument('--train_percs', nargs='+', type=float, default=[0.01, 0.03, 0.1, 0.3, 0.5, 0.8],
                        help='Percentages of data to use for training')
    parser.add_argument('--all_subjects', type=strtobool, nargs='?', const=True, default=False)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--val_perc', type=float, default=0.1)
    parser.add_argument('--subjects', nargs='+', type=str, default=['01'])
    parser.add_argument('--sessions', nargs='+', type=str, default=['0'])
    parser.add_argument('--tasks', nargs='+', type=str, default=['0', '1', '2', '3'])
    parser.add_argument('--val_sessions', nargs='+', type=str, default=None)
    parser.add_argument('--val_tasks', nargs='+', type=str, default=None)
    parser.add_argument('--test_sessions', nargs='+', type=str, default=None)
    parser.add_argument('--test_tasks', nargs='+', type=str, default=None)
    parser.add_argument('--excl_subj_sess_tasks', type=subj_sess_tasks_triple, nargs='+', default=None)
    parser.add_argument('--normalise_data', type=strtobool, nargs='?', const=True, default=True)
    parser.add_argument('--right_context', type=int, default=0)
    parser.add_argument('--left_context', type=int, default=0)
    parser.add_argument('--data_space', type=str, default='source',
                        choices=['source', 'sensor', 'shared_source', 'parcel'])
    parser.add_argument('--ignore_structurals', type=strtobool, nargs='?', const=True, default=False)
    parser.add_argument('--other_subjs', nargs='+', type=str, default=[])
    parser.add_argument('--drop_regions', nargs='+', type=str, default=None)
    parser.add_argument('--clamp', type=float, default=None)
    parser.add_argument('--max_n_sensors', type=int, default=None)
    parser.add_argument('--subj_ind_offset', type=int, default=0)
    parser.add_argument('--pca_voxels', type=int, default=None)
    parser.add_argument('--morph_subj', type=str, default=None)

    # training optimisations
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--prefetch_factor', type=int, default=None)
    parser.add_argument('--test_batch_size', type=int, default=2048)

    # hyperparameters
    parser.add_argument("--model", type=str, default='MLP')
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout_prob', type=float, default=0.)
    parser.add_argument('--extra_layer', type=strtobool, nargs='?', const=True, default=False)
    parser.add_argument("--n_repeats", type=int, default=1)

    # data augmentations
    parser.add_argument('--spatial_dropout_prob', type=float, default=0.0)
    parser.add_argument('--spatial_noise_lenscale', type=float, default=0.0)
    parser.add_argument('--mixup_alpha', type=float, default=-1.)
    parser.add_argument('--vol_aug', type=str, default=None)
    parser.add_argument('--input_dropout_prob', type=float, default=0.0)
    parser.add_argument('--slice_dropout_prob', type=float, default=0.0)
    parser.add_argument('--cb_dropout_prob', type=float, default=0.0)

    # caching
    parser.add_argument("--res_path", type=str, default='res.txt')
    parser.add_argument('--cache_dir', type=str, default=None)

    return parser
