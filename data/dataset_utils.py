import ast
import os
import random
import warnings
from pathlib import Path
from typing import Optional, Type

import mne.io
import numpy as np
import pandas as pd
import torch
from mne import VolSourceEstimate
from mne_bids import BIDSPath
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import Subset

from consts import SOURCE_VOL_SPACING, is_windows
from data.data_utils import get_subject_voxels, get_subject_raw, get_subject_parcels
from data.path_utils import PathArgs
from data.schoffelen_events_utils import read_events_file_schoffelen
from data.structurals_utils import StructuralsData

ARPABET = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
ARPABET_NEGVOICE = ["P", "T", "CH", "F", "TH", "S", "SH", "HH"]


def normalise_data(data):
    """
    Normalize data with 2 or 3 dimensions (n_channels, n_times, [additional_dims]).
    Handles cases where standard deviation might be zero by adding a small epsilon.
    """
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)

    # todo handle case where std is 0, likely in parcels with single voxels

    # Normalize data
    normalise_data = (data - means) / stds
    normalise_data[stds[:, 0] == 0] = 0

    assert not np.isnan(normalise_data).any(), 'NaNs in the normalised data'
    assert not np.isinf(normalise_data).any(), 'Infs in the normalised data'

    assert np.isclose(normalise_data.mean(axis=1), 0, atol=1e-4).all(), 'Data not zero mean'

    return normalise_data


def pca_voxels(data: np.ndarray, n_components: int) -> np.ndarray:
    # data: [n_voxels, 3, n_times]
    n_voxels, n_channels, n_times = data.shape
    assert n_channels == 3, "Input data must have 3 channels per voxel."

    # Initialize the output array for the transformed data
    transformed_data = np.zeros((n_voxels, n_components, n_times))

    for i in range(n_voxels):
        # Extract the data for the current voxel: [3, n_times]
        voxel_data = data[i]  # Shape: [3, n_times]

        # Apply PCA on the transposed data: [n_times, 3] -> [n_times, n_components]
        pca = PCA(n_components=n_components)
        transformed_voxel = pca.fit_transform(voxel_data.T)  # Shape: [n_times, n_components]

        # Transpose back to have shape [n_components, n_times]
        transformed_data[i] = transformed_voxel.T

    return transformed_data


class NeuroSubloader(Dataset):
    def __init__(self, path_args: PathArgs, subject_ind: int, args, labels_type: str = 'speech', cache: bool = True,
                 recalculate: bool = False):
        self.subject_ind = subject_ind if path_args.subj not in args.other_subjs else -1

        if args.data_space == 'sensor':
            activity = get_subject_raw(path_args, read_subj_raw=self.read_subj_raw, cache=cache,
                                       recalculate=recalculate)
            data = activity.get_data(picks='meg')
            sfreq = activity.info['sfreq']

            if args.max_n_sensors is not None:
                # relevant for schoffelen where some subjects have 273 and some have 274 sensors
                assert args.max_n_sensors <= data.shape[0], 'Cannot have more sensors than data'
                data = data[:args.max_n_sensors]
        elif args.data_space == 'source' or args.data_space == 'shared_source':
            morph = args.morph_subj if args.morph_subj is not None \
                else 'fsaverage' if args.data_space == 'shared_source' else None
            structurals = self.get_structurals(path_args, verbose=False) if not args.ignore_structurals else None
            activity = get_subject_voxels(path_args, read_subj_raw=self.read_subj_raw, structurals_data=structurals,
                                          cache=cache, recalculate=recalculate, morph=morph)
            data = activity.data
            sfreq = activity.sfreq

            # vector source estimate, shape (n_voxels, 3, n_times). if models wanna treat each vector
            # component separately they can reshape it internally
            if data.ndim == 3:
                if args.pca_voxels is not None:
                    data = pca_voxels(data, args.pca_voxels)

                data = data.reshape(-1, data.shape[-1])
        elif args.data_space == 'parcel':
            structurals = self.get_structurals(path_args, verbose=False) if not args.ignore_structurals else None
            activity = get_subject_parcels(path_args, read_subj_raw=self.read_subj_raw, structurals_data=structurals,
                                           cache=cache, recalculate=recalculate)
            data = activity.data  # shape: (n_parcels*n_stats*3, n_times)
            sfreq = activity.sfreq
        else:
            raise ValueError(f'Unsupported data space {args.data_space}')

        if args.normalise_data:
            orig_dtype = data.dtype
            dtype = np.float128 if not is_windows else np.float64
            data = data.astype(dtype)
            data = normalise_data(data)
            data = data.astype(orig_dtype)

        if args.clamp is not None:
            assert args.clamp > 0, 'Clamp must be positive'
            data = np.clip(data, -args.clamp, args.clamp)

        if args.drop_regions is not None:
            assert args.data_space == 'source', 'Can only drop regions in source space'
            data = self.drop_regions(data, activity, args.drop_regions)

        assert not np.isnan(data).any(), 'NaNs in the data'
        assert not np.isinf(data).any(), 'Infs in the data'

        assert labels_type == 'speech', 'Still need to properly implement voicing, if you want go for it'
        self.labels = self.get_labels(path_args, labels=labels_type, sfreq=sfreq, n_ts=data.shape[1])

        self.times = activity.times

        assert len(self.labels) == self.data.shape[1] == len(self.times), 'Data and labels must have the same length'

        self.left_context = args.left_context
        self.right_context = args.right_context

        # Compute the valid range for central indices
        self.valid_start_idx = self.left_context
        self.valid_end_idx = len(self.labels) - self.right_context

    def __len__(self):
        # The length is the number of valid central indices
        return self.valid_end_idx - self.valid_start_idx

    def __getitem__(self, idx):
        # Shift index by the valid_start_idx to account for the left context
        central_idx = idx + self.valid_start_idx

        start_idx = central_idx - self.left_context
        end_idx = central_idx + self.right_context + 1

        data = {}
        data['x'] = self.data[:, start_idx:end_idx].astype(np.float32)
        data['subject_inds'] = torch.tensor([self.subject_ind], dtype=torch.long)
        data['y'] = self.labels[central_idx].astype(np.float32)
        data['t'] = self.times[start_idx:end_idx].astype(np.float32)
        return data

    def drop_regions(self, data: np.ndarray, stc: VolSourceEstimate, drop_regions: list[str]) -> np.ndarray:
        voxel_region_map = get_voxel_region_map()
        for region in drop_regions:
            drop_voxel_coords = ...
            raise NotImplementedError('Need to implement dropping regions for ablations')

    @classmethod
    def get_labels(cls, path_args: PathArgs, sfreq: float, n_ts: int, labels: str):
        raise NotImplementedError('Need to implement get_labels method')

    @classmethod
    def read_subj_raw(cls, path_args: PathArgs, verbose: bool = False) -> mne.io.Raw:
        raise NotImplementedError('Need to implement read_subj_raw method')

    @classmethod
    def get_structurals(cls, path_args: PathArgs, verbose: bool = False) -> StructuralsData:
        raise NotImplementedError('Need to implement get_structurals method')


class GWilliamsSubloader(NeuroSubloader):
    def __init__(self, path_args: PathArgs, subject_ind: int, args, labels_type: str = 'speech', cache: bool = True,
                 recalculate: bool = False):
        super().__init__(path_args, subject_ind, args, labels_type, cache, recalculate)

    @classmethod
    def get_speech_labels(
            cls,
            events,
            sfreq: float,
            n_ts: int,
            offset: float = 0.,
    ) -> np.ndarray:
        """Use events to determine speech labels."""

        word_events = events[
            ["'kind': 'word'" in trial_type for trial_type in list(events["trial_type"])]
        ]
        labels = np.zeros(n_ts)
        for i, word_event in word_events.iterrows():
            onset = float(word_event["onset"])
            duration = float(word_event["duration"])
            t_start = int(round((onset + offset) * sfreq))
            t_end = int((onset + offset + duration) * sfreq)
            labels[t_start: t_end + 1] = 1.0

        return labels

    @classmethod
    def get_voicing_labels(
            cls,
            events,
            sfreq: float,
            phoneme_codes,
            offset: float = 0.,
    ) -> tuple[list[int], list[float]]:
        """Use gwilliams events to determine aligned phoneme onsets and their voicing labels.
        Different datasets require different loading functions."""

        offset_samples = int(sfreq * offset)

        # Filter events with phoneme labels
        phoneme_events = events[
            ["'kind': 'phoneme'" in trial_type for trial_type in list(events["trial_type"])]
        ]

        phoneme_onsets = []
        labels = []

        for i, phoneme_event in phoneme_events.iterrows():
            trial_type = ast.literal_eval(phoneme_event["trial_type"])

            phoneme = trial_type["phoneme"].split("_")[0]  # Remove BIE indicators
            onset_samples = int(float(phoneme_event["onset"]) * sfreq) + offset_samples
            phonation = phoneme_codes[phoneme_codes["phoneme"] == phoneme]["phonation"].item()

            # Label as voiced or unvoiced
            if phonation == "v":
                labels.append(1.0)
                phoneme_onsets.append(onset_samples)
            elif phonation == "uv":
                labels.append(0.0)
                phoneme_onsets.append(onset_samples)

        return phoneme_onsets, labels

    @classmethod
    def get_labels(cls, path_args: PathArgs, sfreq: float, n_ts: int, labels: str):
        events = read_events_file(path_args)
        if labels == 'voicing':
            phoneme_codes = pd.read_csv(Path(path_args.data_path) / "phoneme_info.csv")
            return cls.get_voicing_labels(events, sfreq, phoneme_codes)
        elif labels == 'speech':
            return cls.get_speech_labels(events, sfreq, n_ts=n_ts)
        raise ValueError(f'Unsupported label type {labels}')

    @classmethod
    def read_subj_raw(cls, path_args: PathArgs, verbose: bool = False) -> mne.io.Raw:
        bids_path = BIDSPath(subject=path_args.subj, session=path_args.sess, task=path_args.task, root=path_args.root)
        hsp_path = bids_path.copy().update(suffix='headshape', extension='.pos', acquisition='HSP', datatype='meg',
                                           task=None)
        assert os.path.exists(hsp_path.fpath)
        hsp = np.loadtxt(hsp_path.fpath, comments="%")
        hsp /= 1000.0  # m->mm

        elp_path = hsp_path.copy().update(acquisition='ELP').fpath
        assert os.path.exists(elp_path)
        elp = np.loadtxt(elp_path, comments="%")
        elp /= 1000.0  # m->mm

        mrk_path = bids_path.copy().update(suffix='markers', extension='.mrk', datatype='meg').fpath
        # no need to preload, already done when resampling
        raw = mne.io.read_raw_kit(bids_path.fpath, mrk=mrk_path, elp=elp, hsp=hsp, preload=True, verbose=verbose)

        return raw

    @classmethod
    def get_structurals(cls, path_args: PathArgs, verbose: bool = False):
        return None


class RadboudSubloader(NeuroSubloader):
    structurals = {}

    @classmethod
    def read_subj_raw(cls, path_args: PathArgs, verbose: bool = False) -> mne.io.Raw:
        bids_path = BIDSPath(subject=path_args.subj, session=path_args.sess, task=path_args.task, root=path_args.root)
        raw = mne.io.read_raw_ctf(bids_path.fpath, preload=True, verbose=verbose)
        raw.pick(picks='mag')
        return raw

    @classmethod
    def get_structurals(cls, path_args: PathArgs, verbose: bool = False) -> StructuralsData:
        if path_args.subj in cls.structurals:
            return cls.structurals[path_args.subj]

        def constructor():
            subj_dir = path_args.root
            subj = f'sub-{path_args.subj}'

            model = mne.make_bem_model(subject=subj, conductivity=(0.3,), subjects_dir=subj_dir)
            bem = mne.make_bem_solution(model)

            src = mne.setup_volume_source_space(subj, subjects_dir=subj_dir, bem=bem,
                                                pos=SOURCE_VOL_SPACING, verbose=verbose)

            fiducials = mne.coreg.get_mni_fiducials(subject=subj, subjects_dir=subj_dir, verbose=verbose)

            structurals = StructuralsData.create(fiducials=fiducials, src=src, bem=bem)
            return structurals

        cls.structurals[path_args.subj] = StructuralsData(constructor=constructor)

        return cls.structurals[path_args.subj]


class ArmeniSubloader(RadboudSubloader):
    def __init__(self, path_args: PathArgs, subject_ind: int, args, labels_type: str = 'speech', cache: bool = True,
                 recalculate: bool = False):
        super().__init__(path_args, subject_ind, args, labels_type, cache, recalculate)

    @classmethod
    def get_speech_labels(
            cls,
            events,
            sfreq: float,
            n_ts: int,
            offset: float = 0.,
    ) -> np.ndarray:
        """Use events to determine speech labels."""

        phoneme_events = events[["word_onset" in c for c in list(events["type"])]]
        labels = np.zeros(n_ts)
        for i, phoneme_event in phoneme_events.iterrows():
            onset = float(phoneme_event["onset"])
            duration = float(phoneme_event["duration"])
            t_start = int(round((onset + offset) * sfreq))
            t_end = int(round((onset + offset + duration) * sfreq))

            labels[t_start: t_end + 1] = 0.0 if phoneme_event["value"] == "sp" else 1.0
            assert not np.isnan(labels).any()

        return labels

    @classmethod
    def get_labels(cls, path_args: PathArgs, sfreq: float, n_ts: int, labels: str):
        events = read_events_file(path_args)
        if labels == 'speech':
            return cls.get_speech_labels(events, sfreq, n_ts=n_ts)
        raise ValueError(f'Unsupported label type {labels}')


class SchoffelenSubloader(RadboudSubloader):
    def __init__(self, path_args: PathArgs, subject_ind: int, args, labels_type: str = 'speech', cache: bool = True,
                 recalculate: bool = False):
        super().__init__(path_args, subject_ind, args, labels_type, cache, recalculate)

    @classmethod
    def get_speech_labels(
            cls,
            events: pd.DataFrame,
            sfreq: float,
            n_ts: int,
            offset: float = 0.,
    ) -> np.ndarray:
        """Use events to determine speech labels."""

        phoneme_events = events[events.kind == 'phoneme']
        labels = np.zeros(n_ts)
        for i, phoneme_event in phoneme_events.iterrows():
            onset = float(phoneme_event["start"])
            duration = float(phoneme_event["duration"])
            t_start = int(round((onset + offset) * sfreq))
            t_end = int(round((onset + offset + duration) * sfreq))

            labels[t_start: t_end + 1] = 1.
            assert not np.isnan(labels).any()

        return labels

    @classmethod
    def get_labels(cls, path_args: PathArgs, sfreq: float, n_ts: int, labels: str):
        events = read_events_file_schoffelen(path_args, sfreq)
        if labels == 'speech':
            return cls.get_speech_labels(events, sfreq, n_ts=n_ts)
        raise ValueError(f'Unsupported label type {labels}')


class NeuroDataset(Dataset):
    def __init__(self, Subloader: Type[NeuroSubloader], subjects: list[str], sessions: list[str], tasks: list[str],
                 args):
        datasets = []

        found_subjects = set()
        exclude_subj_sess_task = args.excl_subj_sess_tasks
        for subj_ind, subject in enumerate(subjects):
            for session in sessions:
                for task in tasks:
                    if exclude_subj_sess_task is not None and (subject, session, task) in exclude_subj_sess_task:
                        continue

                    path_args = PathArgs(subject, session, task, args.data_path, args.dataset)
                    try:
                        print(f"Loading data for {subject=} {session=} {task=}")
                        datasets.append(Subloader(path_args, subject_ind=subj_ind + args.subj_ind_offset, args=args,
                                                  cache=args.cache_data))
                        found_subjects.add(subject)
                    except FileNotFoundError as e:
                        warnings.warn(f"Data for {subject=} {session=} {task=} not found, skipping. Got error {e}")

        self.datasets = ConcatDataset(datasets)
        self.n_subjects = len(found_subjects)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets.__getitem__(idx)


class GWilliams(NeuroDataset):
    def __init__(self, subjects: list[str], sessions: list[str], tasks: list[str], args):
        super().__init__(GWilliamsSubloader, subjects, sessions, tasks, args)


class Armeni(NeuroDataset):
    def __init__(self, subjects: list[str], sessions: list[str], tasks: list[str], args):
        super().__init__(ArmeniSubloader, subjects, sessions, tasks, args)


class Schoffelen(NeuroDataset):
    def __init__(self, subjects: list[str], sessions: list[str], tasks: list[str], args):
        super().__init__(SchoffelenSubloader, subjects, sessions, tasks, args)


def random_split(dataset, args, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, num_strata=250):
    # uses stratified sampling to ensure there's no data leakage

    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    # Get the total number of samples
    total_samples = len(dataset)

    # Calculate the context size (left + right + 1 for the central sample)
    context_size = args.left_context + args.right_context + 1

    # Calculate the size of each stratum
    stratum_size = total_samples // num_strata

    # Create strata
    strata = [list(range(i * stratum_size, min((i + 1) * stratum_size - context_size, total_samples))) for i in
              range(num_strata)]

    # Allocate samples from each stratum to train, val, and test
    data_inds = {'train': [], 'val': [], 'test': []}
    for stratum in strata:
        # approximate due to context size - we're willing to throw away a tiny bit of data to ensure we prevent leakage
        stratum_dset_sizes = {'train': int(train_ratio * len(stratum)),
                              'val': int(val_ratio * len(stratum)),
                              'test': int(test_ratio * len(stratum))}

        for split in random.sample(['train', 'val', 'test'], 3):
            stratum_split_size = stratum_dset_sizes[split]
            data_inds[split].extend(stratum[:stratum_split_size])
            stratum = stratum[stratum_split_size + context_size:]

    # Create Subset datasets
    train_dataset = Subset(dataset, data_inds['train'])
    val_dataset = Subset(dataset, data_inds['val'])
    test_dataset = Subset(dataset, data_inds['test'])

    return train_dataset, val_dataset, test_dataset


def read_events_file(path_args: PathArgs):
    base_path = Path(path_args.root) / f"sub-{path_args.subj}"
    if path_args.sess is not None:
        base_path = base_path / f"ses-{path_args.sess}"
        return pd.read_csv(
            base_path / 'meg' / f"sub-{path_args.subj}_ses-{path_args.sess}_task-{path_args.task}_events.tsv",
            sep="\t")
    return pd.read_csv(base_path / 'meg' / f"sub-{path_args.subj}_task-{path_args.task}_events.tsv", sep="\t")


def get_phoneme(description: str) -> Optional[str]:
    """Take a phoneme_onset descriptor (e.g. DH0) and returns the ARPABET descriptor."""

    if description in ARPABET:
        return description
    elif (
            len(description) == 3
            and description[2].isnumeric()
            and description[:2] in ARPABET
    ):
        return description[:2]
    else:
        return None


def get_dataset(dataset, subjects, sessions, tasks, args):
    if dataset == 'gwilliams':
        return GWilliams(subjects, sessions, tasks, args=args)
    elif dataset == 'armeni':
        return Armeni(subjects, sessions, tasks, args=args)
    elif dataset == 'schoffelen':
        return Schoffelen(subjects, sessions, tasks, args=args)

    raise ValueError(f'Unsupported dataset {dataset}')


def get_data(args) -> Dataset:
    subjects, sessions, tasks = args.subjects, args.sessions, args.tasks

    print(f"Loading data for {subjects=} {sessions=} {tasks=}")

    return get_dataset(args.dataset, subjects, sessions, tasks, args)
