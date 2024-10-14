import os
from distutils.util import strtobool

import numpy as np
import wandb
from lightning_fabric import seed_everything
from torch import nn
from torch.utils.data import DataLoader

from data.dataset_utils import get_data
from train.models import get_model
from train.parser_utils import get_base_parser
from train.training_utils import get_trainer, SpeechDetectionModel


def get_run_name_base(args):
    """Generate the base run name from args."""
    return (f"{args.model}_ds{args.data_space}_dp{args.dropout_prob}_lr{args.lr}_bs{args.batch_size}"
            f"_epchs{args.num_epochs}_wd{args.weight_decay}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_best_hidden_dim(args, Model: nn.Module, input_dim: int, n_subjs: int) -> int:
    target_params = args.n_params
    low, high = 1, 100
    best_hidden_dim = 0
    closest_diff = float('inf')

    # find sensible upper bound
    args.hidden_dim = high
    while count_parameters(Model(input_dim, args, n_subjects=n_subjs)) < target_params:
        high *= 2
        args.hidden_dim = high

    while low <= high:
        mid = (low + high) // 2

        args.hidden_dim = mid
        model = Model(input_dim, args, n_subjects=n_subjs)
        num_params = count_parameters(model)
        diff = abs(num_params - target_params)

        if diff < closest_diff:
            closest_diff = diff
            best_hidden_dim = mid

        if num_params < target_params:
            low = mid + 1
        elif num_params > target_params:
            high = mid - 1
        else:
            return mid  # Exact match found

    return best_hidden_dim


def train_model(args, test: bool = False) -> tuple:
    train_accs = []
    test_accs = []

    if args.all_subjects:
        subjs = ['all']
        n_subjs = {'armeni': 3}[args.dataset]
    elif args.multi_subj:
        subjs = [args.subjects]
        n_subjs = len(args.subjects)
    else:
        subjs = args.subjects
        n_subjs = 1

    for subj in subjs:
        for _ in range(args.n_repeats):
            print(f"Training with {subj=}")
            train_loader, test_loader = get_dloaders(args)

            # need to have this here as different subjects can have a different # voxels
            input_dim = train_loader.dataset[0]['x'].shape[0]
            Model = get_model(args)
            args.hidden_dim = find_best_hidden_dim(args, Model, input_dim, n_subjs)

            print(f'Hidden dim: {args.hidden_dim}')
            print(f'Desired number of parameters: {args.n_params}')
            n_params = count_parameters(Model(input_dim, args, n_subjects=n_subjs))
            print(f'Actual number of parameters: {n_params}')

            # train a model
            model = SpeechDetectionModel(input_dim, args=args, n_subjects=n_subjs)
            trainer = get_trainer(args, log_model=test)
            trainer.fit(model, train_loader, test_loader)

            train_acc = trainer.test(model, train_loader, ckpt_path='best')[0]['test_acc']
            test_acc = trainer.test(model, test_loader, ckpt_path='best')[0]['test_acc']

            train_accs.append(train_acc)
            test_accs.append(test_acc)

    return train_accs, test_accs


def get_ar_data(args) -> tuple:
    ARMENI_TASK = 'compr'
    args.subjects = ['001']
    args.sessions = ['001']
    args.tasks = [ARMENI_TASK]
    train_set = get_data(args)

    args.sessions = ['002']
    test_set = get_data(args)

    return train_set, test_set


def get_sch_data(args) -> tuple:
    args.subjects = ['A2002', 'A2003']
    args.sessions = [None]
    args.tasks = ['auditory']
    train_set = get_data(args)

    args.subjects = ['A2004', 'A2005']
    args.other_subjs = args.subjects
    test_set = get_data(args)

    return train_set, test_set


def get_dloaders(args) -> tuple:
    train_set, test_set = get_sch_data(args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                              pin_memory=True, prefetch_factor=args.prefetch_factor)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_workers,
                             pin_memory=True, prefetch_factor=args.prefetch_factor)

    return train_loader, test_loader


def train_best_model(args):
    wandb.login(key=os.environ.get('WANDB_API_KEY', None))

    run_name = f"{get_run_name_base(args)}_best_model_eval"
    with wandb.init(project='da_brain', name=run_name, group='best_model_eval'):
        train_accs, test_accs = train_model(args, test=True)

        # Log metrics to wandb
        wandb.log({
            "train_acc": np.mean(train_accs),
            "train_acc_std": np.std(train_accs),
            "test_acc": np.mean(test_accs),
            "test_acc_std": np.std(test_accs),
        })

        print(f"Mean test accuracy: {np.mean(test_accs):.4f}")
        print(f"Test accuracy std:  {np.std(test_accs):.4f}")
        print(f"Test accuracies: {test_accs}")


if __name__ == '__main__':
    parser = get_base_parser()

    # desired # parameters
    parser.add_argument('--n_params', type=int, default=int(0.5e6))

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--multi_subj', type=strtobool, nargs='?', const=True, default=False)

    args = parser.parse_args()

    seed_everything(args.seed)

    train_best_model(args)
