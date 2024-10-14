from distutils.util import strtobool
from itertools import product
from pathlib import Path

import mne
import numpy as np
from mne import make_forward_solution
from mne.beamformer import make_lcmv, apply_lcmv_raw
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from data.data_utils import get_subject_raw
from data.structurals_utils import StructuralsData
from data.dataset_utils import ArmeniSubloader, GWilliamsSubloader
from data.path_utils import PathArgs
from train.parser_utils import get_base_parser


def source_recon(path_args: PathArgs, raw: mne.io.Raw, method: str, voxel_size: int, snr: float,
                 noise: str, pick_ori: str, use_existing_structurals: bool = True,
                 highpass_freq: float = None) -> mne.VectorSourceEstimate:
    subj = f'sub-{path_args.subj}'
    subj_dir = path_args.root

    ignore_ref = path_args.dataset == 'gwilliams'
    if path_args.dataset == 'armeni' and use_existing_structurals:
        model = mne.make_bem_model(subject=subj, conductivity=(0.3,), subjects_dir=subj_dir, verbose=False)
        bem = mne.make_bem_solution(model)

        src = mne.setup_volume_source_space(subj, subjects_dir=subj_dir, bem=bem,
                                            pos=voxel_size, verbose=False)

        fiducials = mne.coreg.get_mni_fiducials(subject=subj, subjects_dir=subj_dir, verbose=False)
    elif path_args.dataset == 'gwilliams' or not use_existing_structurals:
        structurals = StructuralsData.get_def_structural(subjects_dir=path_args.data_path, verbose=False)
        src = structurals.src
        bem = structurals.bem
        fiducials = structurals.fiducials
        subj_dir = path_args.data_path
        subj = 'fsaverage'

    coreg = mne.coreg.Coregistration(
        info=raw.info,
        subject=subj,
        subjects_dir=subj_dir,
        fiducials=fiducials,
    )

    # Fit the fiducials
    if path_args.dataset == 'gwilliams' or not use_existing_structurals:
        coreg.set_scale_mode('3-axis')
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(verbose=False)

    # Forward solution
    fwd = make_forward_solution(raw.info, trans=coreg.trans, src=src, bem=bem, eeg=False,
                                meg=True, verbose=False, ignore_ref=ignore_ref)

    if noise == 'regular':
        data_cov = mne.compute_raw_covariance(raw, verbose=False)
    elif noise == 'diagonal':
        data_cov = mne.compute_raw_covariance(raw, verbose=False)
        data_cov['data'] = np.diag(np.diag(data_cov.data))
    elif noise == 'scalar':
        cov = np.var(raw.get_data())
        data_cov = mne.compute_raw_covariance(raw, verbose=False)
        data_cov['data'] = np.eye(data_cov['data'].shape[0], dtype=data_cov['data'].dtype) * cov
    elif noise == 'adhoc':
        data_cov = mne.make_ad_hoc_cov(raw.info, verbose=False)

    if method == 'lcmv':
        filters = make_lcmv(raw.info, fwd, data_cov, verbose=False)
        stc = apply_lcmv_raw(raw, filters, verbose=False)
    else:
        inv_op = make_inverse_operator(raw.info, fwd, data_cov, verbose=False)

        stc = apply_inverse_raw(raw, inv_op, lambda2=1. / snr ** 2, method=method, pick_ori=pick_ori,
                                verbose=False)

    stc.filter(l_freq=highpass_freq, h_freq=None, verbose=False)

    return stc


def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.T)


def check_nans(data, name):
    assert not np.isnan(data).any(), f"NaNs found in {name}"


def fit_and_evaluate_model(train_data, train_labels, test_data, test_labels):
    check_nans(train_data, "training data")
    check_nans(test_data, "test data")

    clf = LogisticRegression(penalty=None, random_state=42)
    clf.fit(train_data, train_labels)

    y_pred = clf.predict(test_data)
    check_nans(y_pred, "predictions")

    accuracy = balanced_accuracy_score(test_labels, y_pred)
    return accuracy


def process_and_evaluate(train_stc, test_stc, train_labels, test_labels):
    train_data = train_stc.data
    test_data = test_stc.data

    train_data = train_data.reshape(-1, train_data.shape[-1])
    test_data = test_data.reshape(-1, test_data.shape[-1])

    train_data = standardize_data(train_data)
    test_data = standardize_data(test_data)
    return fit_and_evaluate_model(train_data, train_labels, test_data, test_labels)


def eval_source_recon(train_path_args: PathArgs, test_path_args: PathArgs, train_raw: mne.io.Raw, test_raw: mne.io.Raw,
                      train_labels, test_labels, use_structurals: bool = True,
                      res_file: str = 'source_recon_results.txt'):
    recon_methods = ['MNE', 'dSPM', 'sLORETA', 'lcmv']
    voxel_sizes = [10, 15, 20]
    snrs = [0.5, 1, 2, 3, 4, 5]
    noise_estimate = ['regular', 'diagonal', 'scalar', 'adhoc']
    highpass_freqs = [None, 0.1, 0.5, 1, 5]
    pick_oris = ['vector', None]

    params = product(recon_methods, voxel_sizes, snrs, noise_estimate, highpass_freqs, pick_oris)
    params = list(params)
    np.random.shuffle(params)

    best_accuracy = 0
    best_params = None

    for method, voxel_size, snr, noise, highpass_freq, pick_ori in params:
        if method == 'lcmv' and snr != min(snrs):
            continue

        train_stc = source_recon(train_path_args, train_raw, use_existing_structurals=use_structurals, method=method,
                                 voxel_size=voxel_size, snr=snr, noise=noise, highpass_freq=highpass_freq,
                                 pick_ori=pick_ori)
        test_stc = source_recon(test_path_args, test_raw, use_existing_structurals=use_structurals, method=method,
                                voxel_size=voxel_size, snr=snr, noise=noise, highpass_freq=highpass_freq,
                                pick_ori=pick_ori)

        accuracy = process_and_evaluate(train_stc, test_stc, train_labels, test_labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (method, voxel_size, snr, noise)

        print(f'Balanced accuracy for {method}, {voxel_size}, {snr}, {noise}, {highpass_freq}, {pick_ori} '
              f'is {accuracy:.3f}')

        openmode = 'w' if not Path(res_file).exists() else 'a'
        with open(res_file, openmode) as f:
            f.write(f'{method}, {voxel_size}, {snr}, {noise}, {highpass_freq}, {accuracy:.3f}\n')

    print(f'Best parameters: {best_params} with balanced accuracy: {best_accuracy:.3f}')


def preprocess_raw(raw, highpass_freq, lowpass_freq, sfreq, notch):
    if notch:
        raw.notch_filter(np.arange(50, 251, 50), verbose=False)
    raw.filter(highpass_freq, lowpass_freq, verbose=False)
    return raw.resample(sfreq, verbose=False)


def eval_raw(train_path_args: PathArgs, test_path_args: PathArgs, read_sraw: callable, get_labels: callable):
    highpass_freqs = [None, 0.1, 0.5, 1, 5]
    lowpass_freqs = [None, 10, 25, 35, 48, 100, 125]
    sfreqs = [100, 150, 200, 250, 300, 350]
    notch_filters = [False, True]

    params = product(highpass_freqs, lowpass_freqs, sfreqs, notch_filters)
    params = list(params)
    np.random.shuffle(params)

    best_accuracy = 0
    best_params = None

    for highpass_freq, lowpass_freq, sfreq, notch in params:
        if ((highpass_freq is not None and lowpass_freq is not None and highpass_freq > lowpass_freq) or
                (lowpass_freq is not None and lowpass_freq * 2 > sfreq)):
            continue

        train_raw = preprocess_raw(read_sraw(train_path_args), highpass_freq, lowpass_freq, sfreq, notch)
        test_raw = preprocess_raw(read_sraw(test_path_args), highpass_freq, lowpass_freq, sfreq, notch)

        train_labels = get_labels(train_path_args, sfreq, n_ts=train_raw.get_data().shape[1], labels='speech')
        test_labels = get_labels(test_path_args, sfreq, n_ts=test_raw.get_data().shape[1], labels='speech')

        train_data = standardize_data(train_raw.get_data())
        test_data = standardize_data(test_raw.get_data())

        accuracy = fit_and_evaluate_model(train_data, train_labels, test_data, test_labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (highpass_freq, lowpass_freq, sfreq, notch)

        print(f'Balanced accuracy for {highpass_freq}, {lowpass_freq}, {sfreq}, {notch} is {accuracy:.3f}')

    print(f'Best parameters: {best_params} with balanced accuracy: {best_accuracy:.3f}')


def main():
    parser = get_base_parser()
    parser.add_argument('--use_structurals', type=strtobool, nargs='?', const=True, default=True)

    args = parser.parse_args()

    if args.dataset == 'armeni':
        train_path_args = PathArgs(subj='001', sess='001', task='compr', dataset=args.dataset, data_path=args.data_path)
        test_path_args = PathArgs(subj='001', sess='002', task='compr', dataset=args.dataset, data_path=args.data_path)
        Subloader = ArmeniSubloader
    elif args.dataset == 'gwilliams':
        train_path_args = PathArgs(subj='01', sess='0', task='3', dataset=args.dataset, data_path=args.data_path)
        test_path_args = PathArgs(subj='01', sess='1', task='3', dataset=args.dataset, data_path=args.data_path)
        Subloader = GWilliamsSubloader
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.data_space == 'source':
        train_raw = get_subject_raw(train_path_args, read_subj_raw=Subloader.read_subj_raw, cache=False, verbose=False)
        test_raw = get_subject_raw(test_path_args, read_subj_raw=Subloader.read_subj_raw, cache=False, verbose=False)
        train_labels = Subloader.get_labels(train_path_args, sfreq=train_raw.info['sfreq'], labels='speech',
                                            n_ts=train_raw.get_data().shape[1])
        test_labels = Subloader.get_labels(test_path_args, sfreq=test_raw.info['sfreq'], labels='speech',
                                           n_ts=test_raw.get_data().shape[1])
        eval_source_recon(train_path_args, test_path_args, train_raw, test_raw, train_labels, test_labels,
                          use_structurals=args.use_structurals)
    else:
        eval_raw(train_path_args, test_path_args, Subloader.read_subj_raw, Subloader.get_labels)


if __name__ == '__main__':
    main()
