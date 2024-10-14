import os
import pickle

import mne
import numpy as np
import torch
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img
from pytorch_lightning import Trainer

from consts import AR_TEST_SESS, SCH_TEST_SUBJS
from data.data_utils import get_subject_voxels
from data.structurals_utils import StructuralsData
from data.dataset_utils import ArmeniSubloader, SchoffelenSubloader, get_data
from data.path_utils import PathArgs
from train.models import get_model
from train.parser_utils import get_base_parser
from train.training_utils import SpeechDetectionModel
from train.train_opt_model import get_dloaders, find_best_hidden_dim


def create_voxel_region_map(stc: mne.VolSourceEstimate, src: mne.SourceSpaces) -> list[set]:
    ho_atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')
    atlas_img = ho_atlas['maps']
    atlas_labels = ho_atlas['labels']

    stc.subject = src._subject  # hack needed due to structural morphing
    stc.data = stc.data * 0 + 1  # allows recognising which voxels are original ones later
    stc_img = stc.as_volume(src, format='nifti2')

    # Resample the atlas to the source image
    atlas_resampled = resample_to_img(atlas_img, stc_img, interpolation='nearest')

    atlas_data = atlas_resampled.get_fdata()
    # no need for the entire time series! take std to find nonzero/constant voxels
    src_data = stc_img.get_fdata()[:, :, :, 0]

    # Get voxel coordinates in the source space
    voxel_coords = np.array(np.nonzero(src_data)).T

    # Function to get 6 neighboring voxels for a given voxel
    def get_neighbors(voxel_coord):
        neighbors = []
        for dim in range(3):  # x, y, z
            for offset in [-1, 1]:  # previous and next voxel along the dimension
                neighbor = voxel_coord.copy()
                neighbor[dim] += offset
                neighbors.append(neighbor)
        return neighbors

    # Initialize a list to store the set of regions for each voxel in voxel_coords
    voxel_to_regions_list = []

    region_names = {idx: label for idx, label in enumerate(atlas_labels)}

    # Map each voxel and its neighbors to the corresponding atlas label(s)
    for voxel in voxel_coords:
        # Initialize a set to store the regions for the current voxel
        regions = set()

        # Get the label(s) for the current voxel
        voxel_label = atlas_data[tuple(voxel)]
        regions.add(voxel_label)

        # Get neighboring voxels and add their labels to the set
        neighbors = get_neighbors(voxel)
        for neighbor in neighbors:
            # Ensure the neighbor is within the valid bounds of the image
            if (0 <= neighbor[0] < atlas_data.shape[0] and
                    0 <= neighbor[1] < atlas_data.shape[1] and
                    0 <= neighbor[2] < atlas_data.shape[2]):
                neighbor_label = atlas_data[tuple(neighbor)]
                regions.add(neighbor_label)

        # Convert the labels to region names
        region_names_set = {region_names[label] for label in regions}

        # Append the set of region names to the list
        voxel_to_regions_list.append(region_names_set)

    return voxel_to_regions_list


def get_voxel_region_map(args, subj: str):
    if args.dataset == 'armeni':
        morph = None
        path_args = PathArgs(subj=subj, sess='001', task='compr', data_path=args.data_path, dataset='armeni')
        structurals = ArmeniSubloader.get_structurals(path_args, verbose=False)
        read_sraw = ArmeniSubloader.read_subj_raw
    elif args.dataset == 'schoffelen':
        morph = 'fsaverage'
        assert args.data_space == 'shared_source', "Shared source space is required for multi-subj (Schoffelen) eval"
        path_args = PathArgs(subj='A2002', sess=None, task='auditory', data_path=args.data_path, dataset='schoffelen')
        structurals = SchoffelenSubloader.get_structurals(path_args, verbose=False)
        read_sraw = SchoffelenSubloader.read_subj_raw

    stc = get_subject_voxels(path_args, read_sraw, structurals, verbose=False, cache=True, morph=morph)

    if args.dataset == 'schoffelen':
        structurals = StructuralsData.get_def_structural(subjects_dir=args.data_path)  # uses morphed structural

    return create_voxel_region_map(stc, structurals.src)


def get_test_dloaders(args, subj_s: str):
    if args.dataset == 'armeni':
        args.subjects = [subj_s]
        args.sessions = [AR_TEST_SESS]
        args.tasks = ['compr']
    elif args.dataset == 'schoffelen':
        args.subjects = subj_s
        args.sessions = [None]
        args.tasks = ['auditory']
        args.other_subjs = subj_s

    test_set = get_data(args)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return test_loader


def eval_models_on_masked_regions(MIN_VOXELS_IN_REGION: int = 5):
    parser = get_base_parser()

    # todo clean this up, unify somehow with train_opt_model
    parser.add_argument('--n_params', type=int, default=int(0.5e6))

    args = parser.parse_args()

    if args.dataset == 'schoffelen':
        args.data_space = 'shared_source'
        args.subjects = SCH_TEST_SUBJS
        args.multi_subj = True
    elif args.dataset == 'armeni':
        args.multi_subj = False
        args.data_space = 'source'

    models_names = os.listdir(args.cache_dir)
    models_names = [model_name for model_name in models_names if model_name.endswith('.pt')]
    subjs_masked_regions_accs = {} if not args.multi_subj else []
    subjects = args.subjects
    for i, model_name in enumerate(models_names):
        if args.multi_subj:
            n_subjects = 5  # A2006-A2010 for training
            subj_s = subjects
        else:
            assert len(subjects) == len(models_names), "Number of subjects must match the number of models"
            n_subjects = 1
            subj_s = subjects[i]

        print(f"Model {i + 1}/{len(models_names)}: {model_name}")

        model_path = os.path.join(args.cache_dir, model_name)

        test_loader = get_test_dloaders(args, subj_s=subj_s)

        input_dim = test_loader.dataset[0]['x'].shape[0]
        Model = get_model(args)
        args.hidden_dim = find_best_hidden_dim(args, Model, input_dim, n_subjs=n_subjects)

        sdm = SpeechDetectionModel(input_dim, args=args, n_subjects=n_subjects)
        sdm.load_state_dict(torch.load(model_path))

        trainer = Trainer()

        voxels_labels = get_voxel_region_map(args, subj_s)

        baseline_test_acc = trainer.test(sdm, test_loader)[0]['test_acc']

        masked_regions_accs = {'baseline': baseline_test_acc}
        regions = set(region for regions in voxels_labels for region in regions)
        for region_name in regions:
            if region_name == 'Background':
                continue

            voxels_in_region_mask = np.array([region_name in v_labels for v_labels in voxels_labels])
            if np.sum(voxels_in_region_mask) < MIN_VOXELS_IN_REGION:
                continue

            sdm.voxel_mask = voxels_in_region_mask
            masked_acc = trainer.test(sdm, test_loader)[0]['test_acc']
            masked_regions_accs[region_name] = masked_acc

        if not args.multi_subj:
            subjs_masked_regions_accs[subj_s] = masked_regions_accs
        else:
            subjs_masked_regions_accs.append(masked_regions_accs)

    res_path = os.path.join(args.cache_dir, 'masked_regions_accs.pkl')
    with open(res_path, 'wb') as f:
        pickle.dump(subjs_masked_regions_accs, f)


if __name__ == '__main__':
    eval_models_on_masked_regions()
