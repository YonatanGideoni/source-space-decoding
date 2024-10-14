import os
from distutils.util import strtobool

import numpy as np
import torch
from torch.utils.data import DataLoader

from consts import AR_TEST_SESS, SCH_TEST_SUBJS
from data.dataset_utils import get_data
from train.models import get_model
from train.parser_utils import get_base_parser
from train.train_opt_model import find_best_hidden_dim
from train.training_utils import SpeechDetectionModel
from train.training_utils import get_trainer


def eval_sch_on_ar(args):
    args.data_space = 'shared_source'
    args.dataset = 'armeni'

    args.multi_subj = True
    args.sessions = [AR_TEST_SESS]
    args.tasks = ['compr']

    n_subjs = 5  # A2006-A2010 for training
    ar_subjs = ['001', '002', '003']
    args.other_subjs = ar_subjs
    subj_test_accs = {}
    for subj in ar_subjs:
        args.subjects = [subj]
        test_set = get_data(args)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_workers,
                                 pin_memory=True, prefetch_factor=args.prefetch_factor)

        input_dim = test_loader.dataset[0]['x'].shape[0]
        Model = get_model(args)
        args.hidden_dim = find_best_hidden_dim(args, Model, input_dim, n_subjs)

        model = SpeechDetectionModel(input_dim, args=args, n_subjects=n_subjs)
        accs = []
        for model_name in os.listdir(args.cache_dir):
            if not model_name.endswith('.pt'):
                continue
            model.load_state_dict(torch.load(os.path.join(args.cache_dir, model_name)))

            trainer = get_trainer(args)

            test_acc = trainer.test(model, test_loader)[0]['test_acc']
            print(f'Test acc for {subj}: {test_acc:.5f}')
            accs.append(test_acc)
        subj_test_accs[subj] = accs

    print(subj_test_accs)


def eval_ar_on_sch(args):
    args.data_space = 'source'
    args.dataset = 'schoffelen'

    args.multi_subj = False
    args.sessions = [None]
    args.tasks = ['auditory']

    n_subjs = 1  # models trained on a single subject
    args.other_subjs = SCH_TEST_SUBJS
    ar_subjs = ['001', '002', '003']
    subj_test_accs = {}
    model_names = os.listdir(args.cache_dir)
    model_names = [model_name for model_name in model_names if model_name.endswith('.pt')]

    args.subjects = SCH_TEST_SUBJS
    args.cache_data = False
    for model_name, ar_subj in zip(model_names, ar_subjs):
        args.morph_subj = ar_subj
        test_set = get_data(args)  # each time for a different armeni subject to morph into
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_workers,
                                 pin_memory=True, prefetch_factor=args.prefetch_factor)

        input_dim = test_loader.dataset[0]['x'].shape[0]
        Model = get_model(args)
        args.hidden_dim = find_best_hidden_dim(args, Model, input_dim, n_subjs)

        model = SpeechDetectionModel(input_dim, args=args, n_subjects=n_subjs)
        model.load_state_dict(torch.load(os.path.join(args.cache_dir, model_name)))

        trainer = get_trainer(args)

        test_acc = trainer.test(model, test_loader)[0]['test_acc']
        print(f'Test acc for {ar_subj}: {test_acc:.5f}')
        subj_test_accs[ar_subj] = test_acc

    print(subj_test_accs)


def eval_schar_on_ar(args):
    args.data_space = 'shared_source'
    args.dataset = 'armeni'
    args.multi_subj = True
    args.sessions = [AR_TEST_SESS]
    args.tasks = ['compr']

    n_subjs = 7  # A2006-A2010+001-002 for training
    ar_subjs = ['003']
    args.subjects = ar_subjs
    args.other_subjs = ar_subjs

    test_set = get_data(args)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_workers,
                             pin_memory=True, prefetch_factor=args.prefetch_factor)

    input_dim = test_loader.dataset[0]['x'].shape[0]
    Model = get_model(args)
    args.hidden_dim = find_best_hidden_dim(args, Model, input_dim, n_subjs)

    model = SpeechDetectionModel(input_dim, args=args, n_subjects=n_subjs)
    accs = []
    for model_name in os.listdir(args.cache_dir):
        if not model_name.endswith('.pt'):
            continue
        model.load_state_dict(torch.load(os.path.join(args.cache_dir, model_name)))

        trainer = get_trainer(args)

        test_acc = trainer.test(model, test_loader)[0]['test_acc']
        print(f'Test acc for {ar_subjs[0]}: {test_acc:.5f}')
        accs.append(test_acc)
    print(f'Test acc: {np.mean(accs):.5f}+-{np.std(accs):.5f}')


def main():
    parser = get_base_parser()

    parser.add_argument('--n_params', type=int, default=int(0.5e6))
    parser.add_argument('--sensor_eval', type=strtobool, nargs='?', const=True, default=False)
    parser.add_argument('--multi_subj', type=strtobool, nargs='?', const=True, default=False)

    args = parser.parse_args()

    if args.sensor_eval:
        raise NotImplementedError

    if args.dataset == 'schoffelen':
        eval_sch_on_ar(args)
    elif args.dataset == 'armeni':
        eval_ar_on_sch(args)
    elif args.dataset == 'schar_combined':
        eval_schar_on_ar(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
