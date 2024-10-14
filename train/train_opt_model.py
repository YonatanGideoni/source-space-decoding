import os
import random
from distutils.util import strtobool

import numpy as np
import torch
import wandb
from lightning_fabric import seed_everything
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from consts import SCH_TEST_SUBJS, SCH_VAL_SUBJS, AR_TEST_SESS, AR_VAL_SESS
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
    val_accs = []
    test_accs = []

    models = []

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
            train_loader, val_loader, test_loader = get_dloaders(args, subj=subj)

            # need to have this here as different subjects can have a different # voxels
            input_dim = train_loader.dataset[0]['x'].shape[0]
            print(f"Input dim: {input_dim}")
            Model = get_model(args)
            args.hidden_dim = find_best_hidden_dim(args, Model, input_dim, n_subjs)

            print(f'Hidden dim: {args.hidden_dim}')
            print(f'Desired number of parameters: {args.n_params}')
            n_params = count_parameters(Model(input_dim, args, n_subjects=n_subjs))
            print(f'Actual number of parameters: {n_params}')

            # train a model
            model = SpeechDetectionModel(input_dim, args=args, n_subjects=n_subjs)
            trainer = get_trainer(args, log_model=test)
            trainer.fit(model, train_loader, val_loader)

            train_acc = trainer.test(model, train_loader, ckpt_path='best')[0]['test_acc']
            val_acc = trainer.test(model, val_loader, ckpt_path='best')[0]['test_acc']
            if test:
                test_acc = trainer.test(model, test_loader, ckpt_path='best')[0]['test_acc']
                test_accs.append(test_acc)

            train_accs.append(train_acc)
            val_accs.append(val_acc)

            models.append(model)

    if test:
        return train_accs, val_accs, test_accs, models
    return train_accs, val_accs, models


def get_gw_data(args, subj: str,
                val_sess: str = '0', val_task: str = '3', test_sess: str = '1', test_task: str = '3') -> tuple:
    if args.all_subjects:
        args.subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                         '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27']
    elif args.multi_subj:
        args.subjects = subj
    else:
        args.subjects = [subj]

    args.tasks = ['0', '1', '2', '3']
    args.sessions = ['0', '1']
    args.excl_subj_sess_tasks = [(subj, val_sess, val_task), (subj, test_sess, test_task)]
    train_set = get_data(args)

    args.sessions = [val_sess]
    args.tasks = [val_task]
    args.excl_subj_sess_tasks = None
    val_set = get_data(args)

    args.sessions = [test_sess]
    args.tasks = [test_task]
    test_set = get_data(args)

    return train_set, val_set, test_set


def get_ar_data(args, subj: str, val_sess: tuple[str] = (AR_VAL_SESS,), test_sess: tuple = (AR_TEST_SESS,)) -> tuple:
    ARMENI_TASK = 'compr'

    if args.all_subjects:
        args.subjects = ['001', '002', '003']
    elif args.multi_subj:
        args.subjects = subj
    else:
        args.subjects = [subj]

    args.sessions = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    args.tasks = [ARMENI_TASK]
    args.excl_subj_sess_tasks = [(s, v_sess, ARMENI_TASK) for v_sess in val_sess for s in args.subjects] + \
                                [(s, t_sess, ARMENI_TASK) for t_sess in test_sess for s in args.subjects]
    train_set = get_data(args)

    args.sessions = val_sess
    args.excl_subj_sess_tasks = None
    val_set = get_data(args)

    args.sessions = test_sess
    test_set = get_data(args)

    return train_set, val_set, test_set


def get_sch_data(args, subj, val_subjs: list[str] = SCH_VAL_SUBJS,
                 test_subjs: list[str] = SCH_TEST_SUBJS) -> tuple:
    SCH_SESS = None
    SCH_TASK = 'auditory'

    if args.multi_subj:
        args.subjects = subj
        assert isinstance(subj, list), "Multiple subjects should be passed as a list"
    else:
        raise ValueError("Schoffelen dataset does not support single subject training")

    args.sessions = [SCH_SESS]
    args.tasks = [SCH_TASK]
    train_set = get_data(args)

    args.subjects = val_subjs
    args.other_subjs = args.subjects
    val_set = get_data(args)

    args.subjects = test_subjs
    args.other_subjs = args.subjects
    test_set = get_data(args)

    return train_set, val_set, test_set


def get_combined_data(args, subjs: list[str]) -> tuple:
    sch_subjects = [s for s in subjs if s.startswith('A')]
    ar_subjects = [s for s in subjs if s.isdigit()]
    assert len(sch_subjects) + len(ar_subjects) == len(subjs), "Subjects should be either Schoffelen or Armeni"

    args.dataset = 'schoffelen'
    args.subjects = sch_subjects
    sch_train, sch_val, sch_test = get_sch_data(args, sch_subjects)

    args.dataset = 'armeni'
    args.subj_ind_offset = len(sch_subjects)
    args.subjects = ar_subjects
    args.sessions = ['001']
    args.tasks = ['compr']
    ar_train = get_data(args)

    train = ConcatDataset([sch_train, ar_train])

    args.subj_ind_offset = 0
    args.dataset = 'schar_combined'
    args.subjects = sch_subjects + ar_subjects

    return train, sch_val, sch_test


def get_dloaders(args, subj: str) -> tuple:
    if args.dataset == 'gwilliams':
        train_set, val_set, test_set = get_gw_data(args, subj)
    elif args.dataset == 'armeni':
        train_set, val_set, test_set = get_ar_data(args, subj)
    elif args.dataset == 'schoffelen':
        train_set, val_set, test_set = get_sch_data(args, subj)
    elif args.dataset == 'schar_combined':
        train_set, val_set, test_set = get_combined_data(args, subj)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                              pin_memory=True, prefetch_factor=args.prefetch_factor)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_workers,
                            pin_memory=True, prefetch_factor=args.prefetch_factor)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_workers,
                             pin_memory=True, prefetch_factor=args.prefetch_factor)

    return train_loader, val_loader, test_loader


def opt_hyps(args):
    # Function to sample from log-uniform distribution
    def log_uniform(low, high):
        return 10 ** np.random.uniform(np.log10(low), np.log10(high))

    hyp_combinations = []
    for _ in range(1000):
        dropout_prob = random.choice(args.dropout_probs)
        lr = log_uniform(args.lr_lower, args.lr_upper)
        batch_size = random.choice(args.batch_sizes)
        num_epochs = random.choice(args.epochs)
        weight_decay = log_uniform(args.weight_decay_lower, args.weight_decay_upper)
        hyp_combinations.append((dropout_prob, lr, batch_size, num_epochs, weight_decay))

    # Shuffle the combinations randomly
    random.shuffle(hyp_combinations)

    assert args.normalise_data, 'Data should be normalised'
    assert args.mixup_alpha == -1., 'Mixup is not used'
    assert not args.vol_aug, 'Volume augmentation is not used'

    # give res file a header
    res_dir = os.path.dirname(args.res_path)
    if res_dir:
        os.makedirs(res_dir, exist_ok=True)
    filemode = 'w' if not os.path.exists(args.res_path) else 'a'
    with open(args.res_path, filemode) as f:
        f.write("dropout_prob, lr, batch_size, num_epochs, weight_decay, train_acc, train_acc_std, val_acc, "
                "val_acc_std, approx_n_params\n")

    best_hyps = None
    best_val_acc = 0
    for hyps in hyp_combinations:
        # set hyperparameters
        args.dropout_prob, args.lr, args.batch_size, args.num_epochs, args.weight_decay = hyps

        run_name = f'{get_run_name_base(args)}_opthyps'
        with wandb.init(project='da_brain', name=run_name, group='opt_hyps'):

            train_accs, val_accs, models = train_model(args, test=False)

            train_acc = np.mean(train_accs)
            train_acc_std = np.std(train_accs)
            val_acc = np.mean(val_accs)
            val_acc_std = np.std(val_accs)

            # Log metrics to wandb
            wandb.log({
                "train_acc": train_acc,
                "train_acc_std": train_acc_std,
                "val_acc": val_acc,
                "val_acc_std": val_acc_std,
                "n_params": args.n_params
            })

        # save the results
        with open(args.res_path, 'a') as f:
            f.write(f"{hyps}, {train_acc:.4f}, {train_acc_std:.4f}, {val_acc:.4f}, {val_acc_std:.4f}, "
                    f"{args.n_params}\n")

        # update the best hyperparameters
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_hyps = hyps

            print(f"New best hyperparameters: {best_hyps}, val acc: {best_val_acc}")

            if args.cache_dir is not None:
                models_dir = os.path.join(args.cache_dir, run_name)
                os.makedirs(models_dir, exist_ok=True)
                for i, model in enumerate(models):
                    model_path = os.path.join(models_dir, f"model_{i}.pt")
                    torch.save(model.state_dict(), model_path)

    print(f"Best hyperparameters: {best_hyps}")


def train_best_model(args):
    wandb.login(key=os.environ.get('WANDB_API_KEY', None))

    run_name = f"{get_run_name_base(args)}_best_model_eval"
    with wandb.init(project='da_brain', name=run_name, group='best_model_eval'):
        train_accs, val_accs, test_accs, models = train_model(args, test=True)

        # Log metrics to wandb
        wandb.log({
            "train_acc": np.mean(train_accs),
            "train_acc_std": np.std(train_accs),
            "val_acc": np.mean(val_accs),
            "val_acc_std": np.std(val_accs),
            "test_acc": np.mean(test_accs),
            "test_acc_std": np.std(test_accs),
        })

        print(f"Mean val accuracy: {np.mean(val_accs):.4f}")
        print(f"Val accuracy std:  {np.std(val_accs):.4f}")
        print(f"Val accuracies: {val_accs}")

        print(f"Mean test accuracy: {np.mean(test_accs):.4f}")
        print(f"Test accuracy std:  {np.std(test_accs):.4f}")
        print(f"Test accuracies: {test_accs}")

        if args.cache_dir is not None:
            os.makedirs(args.cache_dir, exist_ok=True)
            for i, model in enumerate(models):
                model_path = os.path.join(args.cache_dir, f"{run_name}_model_{i}.pt")
                torch.save(model.state_dict(), model_path)
                print(f"Model {i} saved to {model_path}")


if __name__ == '__main__':
    parser = get_base_parser()

    # desired # parameters
    parser.add_argument('--n_params', type=int, default=int(0.5e6))

    # hyps to opt over
    parser.add_argument('--dropout_probs', nargs='+', type=float, default=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument('--lr-upper', type=float, default=1e-3)
    parser.add_argument('--lr-lower', type=float, default=1e-7)
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument('--epochs', nargs='+', type=int, default=[100])
    parser.add_argument('--weight_decay-lower', type=float, default=1e-5)
    parser.add_argument('--weight_decay-upper', nargs='+', type=float, default=3e-1)

    parser.add_argument('--seed', type=int, default=42)

    # opt hyps or run best
    parser.add_argument('--opt_hyps', type=strtobool, nargs='?', const=True, default=True)

    # run single subj exps or multiple
    parser.add_argument('--multi_subj', type=strtobool, nargs='?', const=True, default=False)

    args = parser.parse_args()

    if args.all_subjects:
        args.multi_subj = True

    seed_everything(args.seed)

    if args.opt_hyps:
        opt_hyps(args)
    else:
        train_best_model(args)
