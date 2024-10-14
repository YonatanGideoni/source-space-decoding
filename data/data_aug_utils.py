import os

import mne
import numpy as np
import torch
import torch.nn.functional as F
from mne_bids import BIDSPath
from monai.transforms import RandomizableTransform, Transform
from torch import nn

from consts import device
from data.data_utils import get_subject_voxels
from data.structurals_utils import StructuralsData
from data.dataset_utils import GWilliamsSubloader, ArmeniSubloader, SchoffelenSubloader
from data.path_utils import PathArgs
from train.parser_utils import get_base_parser


def get_vert2coord_path(args) -> str:
    # Get the directory where the current script is located
    dir_path = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(dir_path, '..', 'vert2coord', args.data_space, f'{args.dataset}.npy')

    return file_path


def get_default_vert2coords(args) -> np.ndarray:
    file_path = get_vert2coord_path(args)

    if not os.path.exists(file_path):
        print("vert2coord not found, creating it for all datasets. This may take a while.")
        create_all_dsets_vert2coords(args)

    with open(file_path, 'rb') as f:
        vert2coord = np.load(f)
    return vert2coord


class DataAug(nn.Module):
    def __init__(self, args):
        super(DataAug, self).__init__()
        self.args = args

        self.vert2coords = torch.tensor(get_default_vert2coords(args), dtype=torch.float32, device=device) \
            if 'source' in args.data_space else None

    def spatial_dropout(self, x):
        raise NotImplementedError

    def spatial_noise(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, x, y):
        if self.args.spatial_dropout_prob:
            x = self.spatial_dropout(x)
        if self.args.spatial_noise_lenscale:
            x = self.spatial_noise(x)

        return x, y


def create_vert2coord(voxels: mne.VolVectorSourceEstimate, structurals: StructuralsData):
    # as long as the same structurals are used for all subjects - the fsaverage ones in our case - this will output
    # the same result regardless of the subject. The code has the default ones hardcoded, if you change the default
    # parameters (eg. source space resolution) you will get different results and need to run this again.
    src_spaces = [s for s in structurals.src]
    assert len(src_spaces) == 1
    src_space = src_spaces[0]

    assert len(voxels.vertices) == 1
    vertices = voxels.vertices[0]

    coords = []
    for vert in vertices:
        if vert in src_space['vertno']:
            coords.append(src_space['rr'][vert])
        else:
            raise ValueError(f"Vertex {vert} not in source space")

    assert len(coords) == len(vertices)

    return np.array(coords)


def mixup(data: torch.Tensor, targets: torch.Tensor, alpha: float, n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    # graciously taken from https://github.com/hysts/pytorch_mixup/blob/master/utils.py
    indices = torch.randperm(data.size(0), device=device)
    data2 = data[indices]
    targets2 = targets[indices]

    targets = F.one_hot(targets.long(), n_classes)
    targets2 = F.one_hot(targets2.long(), n_classes)

    lam = torch.tensor([np.random.beta(alpha, alpha)], dtype=torch.float32, device=device)
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


def create_all_dsets_vert2coords(args):
    orig_dset = args.dataset
    datasets = [
        {
            'name': 'gwilliams',
            'path_args': lambda: PathArgs(subj='01', sess='0', task='0', data_path=args.data_path, dataset='gwilliams'),
            'structurals': lambda: StructuralsData.get_def_structural(subjects_dir=args.data_path),
            'read_subj_raw': GWilliamsSubloader.read_subj_raw
        },
        {
            'name': 'armeni',
            'path_args': lambda: PathArgs(subj='001', sess='001', task='compr', data_path=args.data_path,
                                          dataset='armeni'),
            'structurals': lambda path_args: ArmeniSubloader.get_structurals(path_args),
            'read_subj_raw': memory_light_read_subj_raw
        },
        {
            'name': 'schoffelen',
            'path_args': lambda: PathArgs(subj='A2002', sess=None, task='auditory', data_path=args.data_path,
                                          dataset='schoffelen'),
            'structurals': lambda path_args: SchoffelenSubloader.get_structurals(path_args),
            'read_subj_raw': memory_light_read_subj_raw
        }
    ]

    def process_dataset(dataset):
        args.dataset = dataset['name']
        path_args = dataset['path_args']()
        structurals = dataset['structurals'](path_args) \
            if 'path_args' in dataset['structurals'].__code__.co_varnames else dataset['structurals']()

        for dspace in ['source', 'shared_source']:
            args.data_space = dspace
            morph = args.data_space == 'shared_source'

            stc = get_subject_voxels(path_args, read_subj_raw=dataset['read_subj_raw'], structurals_data=structurals,
                                     morph=morph)

            if morph:
                structurals = StructuralsData.get_def_structural(subjects_dir=args.data_path)

            vert2coord = create_vert2coord(stc, structurals)

            v2c_path = get_vert2coord_path(args)
            os.makedirs(os.path.dirname(v2c_path), exist_ok=True)
            with open(v2c_path, 'wb') as f:
                np.save(f, vert2coord)

    for dataset in datasets:
        try:
            process_dataset(dataset)
        except Exception as e:
            print(f"Error processing dataset {dataset['name']}: {str(e)}")

    args.dataset = orig_dset


def memory_light_read_subj_raw(path_args: PathArgs, verbose: bool = False) -> mne.io.Raw:
    bids_path = BIDSPath(subject=path_args.subj, session=path_args.sess, task=path_args.task, root=path_args.root)
    raw = mne.io.read_raw_ctf(bids_path.fpath, preload=True, verbose=verbose)
    raw.crop(0, 20)
    raw.pick(picks='mag')
    return raw


class RandomCubeZeroing(RandomizableTransform):
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        assert isinstance(img, torch.Tensor), "Input must be a torch.Tensor"
        assert img.dim() == 5, "Input must be a 5D tensor with shape [batch, channels, x, y, z]"

        batch_size, n_channels, x, y, z = img.shape

        # Generate two random points for each batch
        point1 = torch.stack([
            torch.randint(0, x, (batch_size,)),
            torch.randint(0, y, (batch_size,)),
            torch.randint(0, z, (batch_size,))
        ], dim=1)

        point2 = torch.stack([
            torch.randint(0, x, (batch_size,)),
            torch.randint(0, y, (batch_size,)),
            torch.randint(0, z, (batch_size,))
        ], dim=1)

        # Calculate the min and max coordinates to define the cube
        mins, _ = torch.min(torch.stack([point1, point2]), dim=0)
        maxs, _ = torch.max(torch.stack([point1, point2]), dim=0)
        mins, maxs = mins.to(device), maxs.to(device)

        # Create coordinate grids
        grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(x), torch.arange(y), torch.arange(z), indexing='ij')
        grid_x, grid_y, grid_z = grid_x.to(device), grid_y.to(device), grid_z.to(device)

        # Create mask tensor
        mask = ((grid_x.unsqueeze(0) >= mins[:, 0].view(-1, 1, 1, 1)) &
                (grid_x.unsqueeze(0) <= maxs[:, 0].view(-1, 1, 1, 1)) &
                (grid_y.unsqueeze(0) >= mins[:, 1].view(-1, 1, 1, 1)) &
                (grid_y.unsqueeze(0) <= maxs[:, 1].view(-1, 1, 1, 1)) &
                (grid_z.unsqueeze(0) >= mins[:, 2].view(-1, 1, 1, 1)) &
                (grid_z.unsqueeze(0) <= maxs[:, 2].view(-1, 1, 1, 1)))

        # Expand mask to match the input shape (including channels)
        mask = mask.unsqueeze(1).expand(-1, n_channels, -1, -1, -1)

        return img * (~mask).float()


class SliceDropout(Transform):
    def __init__(self, prob: float = 0.1):
        super().__init__()
        self.prob = prob
        self.dropout = nn.Dropout3d(prob)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        assert isinstance(img, torch.Tensor), "Input must be a torch.Tensor"
        assert img.dim() == 5, "Input must be a 5D tensor with shape [batch, channels, x, y, z]"

        # Apply dropout along x-axis
        x_dropout = self.dropout(img.transpose(1, 2)).transpose(1, 2)

        # Apply dropout along y-axis
        y_dropout = self.dropout(x_dropout.transpose(1, 3)).transpose(1, 3)

        # Apply dropout along z-axis
        z_dropout = self.dropout(y_dropout.transpose(1, 4)).transpose(1, 4)

        return z_dropout
