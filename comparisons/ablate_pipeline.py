import sys
from copy import deepcopy

import mne
import numpy as np
from mne.beamformer import apply_lcmv_raw, make_lcmv
from mne.minimum_norm import apply_inverse_raw, make_inverse_operator
from torch.utils.data import DataLoader

from consts import DEF_DATA_PATH
from data.path_utils import PathArgs


def get_subject_fwd_sol(path_args: PathArgs, read_subj_raw: callable, structurals_data=None,
                        verbose: bool = False) -> tuple[mne.io.Raw, mne.Forward]:
    raw = get_subject_raw(path_args, read_subj_raw=read_subj_raw, verbose=verbose)

    # Coregistration
    allow_scaling, ignore_ref = False, False
    if structurals_data is None:
        from data.structurals_utils import StructuralsData

        allow_scaling, ignore_ref = True, True
        structurals_data = StructuralsData.get_def_structural(path_args.data_path, verbose=verbose)
        structurals_data.set_subject_fiducials(path_args, verbose=verbose)

        subj = 'fsaverage'
        subj_dir = path_args.data_path
    else:
        subj = f'sub-{path_args.subj}'
        subj_dir = path_args.root

    coreg = mne.coreg.Coregistration(
        info=raw.info,
        subject=subj,
        subjects_dir=subj_dir,
        fiducials=structurals_data.fiducials,
    )
    if allow_scaling:
        coreg.set_scale_mode('3-axis')  # needed because fsaverage is used instead of real structurals
    coreg.fit_fiducials(verbose=verbose)
    coreg.fit_icp(verbose=verbose)

    # Compute the forward solution. ignore ref is needed for KIT data
    fwd = mne.make_forward_solution(raw.info, trans=coreg.trans, src=structurals_data.src, bem=structurals_data.bem,
                                    meg=True, eeg=False, ignore_ref=ignore_ref, verbose=verbose)

    return raw, fwd


def get_subject_raw(path_args: PathArgs, read_subj_raw: callable, sfreq: float = 150., l_freq: float = 0.1,
                    h_freq: float = 48., notch_filter: bool = False, verbose: bool = False, **kwargs) -> mne.io.Raw:
    assert sfreq >= 2 * h_freq, 'Downsampling to higher than Nyquist frequency'

    raw: mne.io.Raw = read_subj_raw(path_args, verbose=verbose)

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    if notch_filter:
        raw.notch_filter(np.arange(50, 251, 50), verbose=verbose)

    # downsample, natively at >=1kHz which is super excessive
    raw = raw.resample(sfreq, verbose=verbose)

    return raw


def get_subject_voxels(path_args: PathArgs, read_subj_raw: callable, structurals_data=None,
                       snr: float = 3, verbose: bool = False, morph: bool = False,
                       pick_ori: str = 'vector', method: str = 'MNE',
                       cov_form: str = 'diagonal', use_structurals: bool = True,
                       **kwargs) -> mne.VolVectorSourceEstimate:
    if not use_structurals:
        structurals_data = None
    raw, fwd = get_subject_fwd_sol(path_args, read_subj_raw, structurals_data, verbose=verbose)

    if cov_form == 'regular':
        data_cov = mne.compute_raw_covariance(raw, verbose=False)
    elif cov_form == 'diagonal':
        data_cov = mne.compute_raw_covariance(raw, verbose=False)
        data_cov['data'] = np.diag(np.diag(data_cov.data))
    elif cov_form == 'scalar':
        cov = np.var(raw.get_data())
        data_cov = mne.compute_raw_covariance(raw, verbose=False)
        data_cov['data'] = np.eye(data_cov['data'].shape[0], dtype=data_cov['data'].dtype) * cov
    elif cov_form == 'adhoc':
        data_cov = mne.make_ad_hoc_cov(raw.info, verbose=False)

    if method == 'lcmv':
        filters = make_lcmv(raw.info, fwd, data_cov, verbose=False)
        stc = apply_lcmv_raw(raw, filters, verbose=False)
    else:
        inv_op = make_inverse_operator(raw.info, fwd, data_cov, verbose=False)

        stc = apply_inverse_raw(raw, inv_op, lambda2=1. / snr ** 2, method=method, pick_ori=pick_ori,
                                verbose=False)

    if morph:
        from data.structurals_utils import morph_stc_to_fsaverage
        stc = morph_stc_to_fsaverage(stc, structurals_data, path_args, verbose=verbose)

    return stc


class MockArgs:
    lr = 1e-5
    batch_size = 128
    weight_decay = 0.
    dropout_prob = 0.
    n_params = int(0.25e6)
    test_batch_size = 2048
    model = 'MLP'
    dataset = 'armeni'
    data_path = DEF_DATA_PATH
    data_space = None
    num_epochs = 100
    excl_subj_sess_tasks = None
    normalise_data = True
    clamp = None
    other_subjs = []
    drop_regions = None
    left_context = 0
    right_context = 0
    ignore_structurals = False
    input_dropout_prob = 0.
    slice_dropout_prob = 0.
    cb_dropout_prob = 0.
    spatial_dropout_prob = 0.
    spatial_noise_lenscale = 0.
    mixup_alpha = -1.
    vol_aug = None
    prefetch_factor = None
    n_workers = 0
    subj_ind_offset = 0
    max_n_sensors = None
    extra_layer = False
    pca_voxels = None
    cache_data = False


def benchmark(params, is_src: bool) -> float:
    train_set, val_set, test_set = get_data_manual(params, is_src)

    args = MockArgs()
    args.data_space = 'source' if is_src else 'sensor'

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

    input_dim = test_loader.dataset[0]['x'].shape[0]
    from train.models import get_model
    Model = get_model(args)

    from train.train_opt_model import find_best_hidden_dim
    args.hidden_dim = find_best_hidden_dim(args, Model, input_dim, n_subjs=1)

    from train.training_utils import SpeechDetectionModel
    model = SpeechDetectionModel(input_dim, args=args, n_subjects=1)
    from train.training_utils import get_trainer
    trainer = get_trainer(args)
    trainer.fit(model, train_loader, val_loader)

    return trainer.test(model, test_loader, ckpt_path='best')[0]['test_acc']


def get_data_manual(params, src: bool):
    args = MockArgs()
    args.data_space = 'source' if src else 'sensor'
    args.pca_voxels = params.pop('pca_voxels', None)

    # monkey patch, monkey patch, does whatever a monkey can. this is why all brainvoxel imports are local
    import data.data_utils
    def modified_get_subject_raw(path_args, read_subj_raw, **kwargs):
        for param_name in params:
            if param_name in kwargs:
                del kwargs[param_name]
        return get_subject_raw(path_args, read_subj_raw, **params, **kwargs)

    def modified_get_subject_voxels(path_args, read_subj_raw, structurals_data, **kwargs):
        for param_name in params:
            if param_name in kwargs:
                del kwargs[param_name]
        return get_subject_voxels(path_args, read_subj_raw, structurals_data, **params, **kwargs)

    data.data_utils.get_subject_raw = modified_get_subject_raw
    data.data_utils.get_subject_voxels = modified_get_subject_voxels

    from data.dataset_utils import get_dataset
    from data.dataset_utils import RadboudSubloader
    RadboudSubloader.structurals = {}
    train_set = get_dataset(args.dataset, subjects=['001'], sessions=['001'], tasks=['compr'], args=args)
    val_set = get_dataset(args.dataset, subjects=['001'], sessions=['002'], tasks=['compr'], args=args)
    test_set = get_dataset(args.dataset, subjects=['001'], sessions=['003'], tasks=['compr'], args=args)

    sys.modules.pop('data.data_utils')  # "unimport" the module

    return train_set, val_set, test_set


def ablate_pipeline(feat_name: str, feat_vals: list, n_repeats: int = 1) -> list[float]:
    DEF_PARAMS_RAW = {'sfreq': 150., 'l_freq': 0.1, 'h_freq': 48., 'notch_filter': False}
    DEF_PARAMS_SRC = {'snr': 3, 'morph': False, 'pick_ori': 'vector', 'method': 'MNE', 'cov_form': 'diagonal',
                      'use_structurals': True, 'pca_voxels': None}
    HFREQ_SFREQ_CORRESPONDENCE = {48.: 150., 60.: 200., 100.: 250., 150.: 350.}

    is_src = True
    if feat_name in DEF_PARAMS_RAW:
        params = deepcopy(DEF_PARAMS_RAW)
        is_src = False
    elif feat_name in DEF_PARAMS_SRC:
        params = deepcopy(DEF_PARAMS_SRC)
    else:
        raise ValueError(f"Feature name {feat_name} not recognised")

    test_accs = []
    for fval in feat_vals:
        print(f"Testing {feat_name}={fval}")
        params[feat_name] = fval
        if feat_name == 'h_freq':
            params['sfreq'] = HFREQ_SFREQ_CORRESPONDENCE[fval]  # ensure >2*nyquist

        for _ in range(n_repeats):
            test_acc = benchmark(params, is_src)
            test_accs.append(test_acc)

            print(f"Test accuracy: {test_acc:.5f} for {feat_name}={fval}")

        print(f"Test accuracy: {np.mean(test_accs):.5f}+-{np.std(test_accs):.5f} for {feat_name}={fval}")

    return test_accs


def main():
    ablate_pipeline('h_freq', [48., 60., 100., 150., None])
    ablate_pipeline('use_structurals', [False, True])
    ablate_pipeline('pick_ori', ['vector', None])
    ablate_pipeline('morph', [False, True])
    # ablate_pipeline('voxel_size', [10., 15., 20.]) todo implement
    ablate_pipeline('snr', [0.5, 1, 3, 5, 10])
    ablate_pipeline('sfreq', [100., 150., 300.])
    ablate_pipeline('l_freq', [None, 0.1, 0.5, 1., 5., 10.])
    ablate_pipeline('cov_form', ['regular', 'diagonal', 'scalar', 'adhoc'])
    ablate_pipeline('method', ['MNE', 'lcmv', 'dSPM', 'sLORETA'])
    ablate_pipeline('notch_filter', [False, True])
    ablate_pipeline('pca_voxels', [None, 1, 2])


if __name__ == '__main__':
    main()
