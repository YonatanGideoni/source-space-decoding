import os

import mne
import numpy as np
from mne.minimum_norm import apply_inverse_raw, make_inverse_operator
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img
from scipy import sparse
from scipy.ndimage import label

from data.structurals_utils import morph_stc_to_subj, StructuralsData
from data.path_utils import PathArgs

PARCEL_STATS: list[callable] = [np.mean, np.std, np.max, np.min]


class VolParcelEstimate:
    def __init__(self, parcel_data, parcel_sizes, adjacency_list, times, sfreq):
        """
        Stores aggregated parcel-wise data along with other relevant information.

        :param parcel_data: A numpy array of shape (n_parcels, n_times, stats, 3)
        :param parcel_sizes: Number of voxels in each parcel (np.ndarray of shape (n_parcels,))
        :param adjacency_list: List of adjacency between parcels (list of tuples)
        :param times: Time points associated with the data (np.ndarray)
        :param sfreq: Sampling frequency of the data (float)
        """
        self.parcel_data = parcel_data  # shape: (n_parcels, n_times, stats, 3)
        self.parcel_sizes = parcel_sizes  # shape: (n_parcels,)
        self.adjacency_list = adjacency_list
        self.times = times
        self.sfreq = sfreq

    def get_parcel_stat(self, stat_idx):
        """Helper function to return a specific statistic for all parcels"""
        return self.data[:, :, stat_idx]

    @property
    def data(self):
        n_ts = self.parcel_data.shape[1]
        return self.parcel_data.reshape(-1, n_ts)


def get_subject_raw(path_args: PathArgs, read_subj_raw: callable, sfreq: float = 150., l_freq: float = 0.1,
                    h_freq: float = 48., cache: bool = False, recalculate: bool = False,
                    verbose: bool = False) -> mne.io.Raw:
    assert sfreq >= 2 * h_freq, 'Downsampling to higher than Nyquist frequency'
    if cache and not recalculate:
        cache_path = path_args.get_raw_cache_path(l_freq is not None)
        if os.path.exists(cache_path):
            return mne.io.read_raw_fif(cache_path)
    raw: mne.io.Raw = read_subj_raw(path_args, verbose=verbose)

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    # downsample, natively at >=1kHz which is super excessive
    raw = raw.resample(sfreq, verbose=verbose)

    if cache:
        cache_path = path_args.get_raw_cache_path(l_freq is not None)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw.save(cache_path, overwrite=True)

    return raw


def get_subject_fwd_sol(path_args: PathArgs, read_subj_raw: callable, structurals_data: StructuralsData = None,
                        verbose: bool = False, cache: bool = False, recalculate: bool = False) -> tuple[
    mne.io.Raw, mne.Forward]:
    raw = get_subject_raw(path_args, read_subj_raw=read_subj_raw, cache=cache, recalculate=recalculate, verbose=verbose)

    # Coregistration
    allow_scaling, ignore_ref = False, False
    if structurals_data is None:
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


def get_subject_voxels(path_args: PathArgs, read_subj_raw: callable, structurals_data: StructuralsData = None,
                       verbose: bool = False, cache: bool = False, recalculate: bool = False,
                       morph: str = None) -> mne.VolVectorSourceEstimate:
    if cache and not recalculate:
        cache_path = path_args.get_voxel_cache_path(morph)
        vol_path = cache_path + '-stc.h5'
        if os.path.exists(vol_path):
            try:
                return mne.read_source_estimate(vol_path)
            except ValueError:
                # happens sometimes with armeni morphed source estimates, no clue why, cryptic mne bug on vertices order
                pass

    raw, fwd = get_subject_fwd_sol(path_args, read_subj_raw, structurals_data, verbose=verbose, cache=cache,
                                   recalculate=recalculate)

    data_cov = mne.compute_raw_covariance(raw, verbose=verbose)
    data_cov['data'] = np.diag(np.diag(data_cov.data))

    inv_op = make_inverse_operator(raw.info, fwd, data_cov, verbose=verbose)

    snr = 3.  # default+empirically usually works best although it seems not to matter much
    stc = apply_inverse_raw(raw, inv_op, lambda2=1. / snr ** 2, method='MNE', pick_ori='vector', verbose=verbose)

    if morph is not None:  # todo make not only fsaverage
        stc = morph_stc_to_subj(stc, structurals_data, path_args, morph, verbose=verbose)

    if cache:
        cache_path = path_args.get_voxel_cache_path(morph)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        stc.save(cache_path, overwrite=True)

    return stc


def compute_parcel_adjacency(atlas_data, n_parcels):
    """
    Compute adjacency between parcels based on spatial proximity.

    :param atlas_data: 3D array containing parcel labels for each voxel
    :param n_parcels: Number of parcels in the atlas
    :return: List of tuples representing adjacent parcels
    """
    adjacency_list = []
    for parcel_id in range(n_parcels):
        parcel_mask = (atlas_data == parcel_id)

        # Label connected regions (adjacent voxels with the same parcel ID)
        labeled_array, num_features = label(parcel_mask)

        for feature_id in range(1, num_features + 1):
            feature_mask = (labeled_array == feature_id)
            for neighbor_id in range(n_parcels):
                if neighbor_id != parcel_id and np.any(atlas_data[feature_mask] == neighbor_id):
                    adjacency_list.append((parcel_id, neighbor_id))

    return list(set(adjacency_list))  # Remove duplicates


def resample_atlas_to_stc(stc, src, atlas_img):
    stc_img = stc.as_volume(src, format='nifti2')
    return resample_to_img(atlas_img, stc_img, interpolation='nearest')


def create_parcel_voxel_matrix(atlas_data):
    atlas_flat = atlas_data.ravel()
    return sparse.csr_matrix((np.ones_like(atlas_flat),
                              (atlas_flat, np.arange(len(atlas_flat)))))


def compute_parcel_statistics(parcel_voxel_matrix, voxel_data, stc_coords, atlas_data):
    """
    Compute parcel statistics with dimension matching.

    :param parcel_voxel_matrix: Sparse matrix mapping parcels to voxels in atlas space
    :param voxel_data: Data for each voxel in the source estimate
    :param stc_coords: Coordinates of non-zero voxels in the source estimate
    :param atlas_data: Resampled atlas data
    :return: Tuple of parcel statistics and parcel sizes
    """
    n_parcels = parcel_voxel_matrix.shape[0]

    # Create a new sparse matrix that maps parcels to voxels in the source estimate space
    rows = []
    cols = []
    for i, coord in enumerate(stc_coords):
        parcel_id = atlas_data[tuple(coord)]
        rows.append(parcel_id)
        cols.append(i)

    data = np.ones_like(rows)
    stc_parcel_voxel_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_parcels, len(stc_coords)))

    parcel_sizes = np.array(stc_parcel_voxel_matrix.sum(axis=1)).flatten()
    parcel_sums = stc_parcel_voxel_matrix.dot(voxel_data)
    parcel_sum_squares = stc_parcel_voxel_matrix.dot(voxel_data ** 2)

    with np.errstate(divide='ignore', invalid='ignore'):
        parcel_means = np.divide(parcel_sums, parcel_sizes[:, np.newaxis])
        parcel_variances = np.divide(parcel_sum_squares, parcel_sizes[:, np.newaxis]) - parcel_means ** 2
        parcel_stds = np.sqrt(np.maximum(parcel_variances, 0))

    parcel_maxs = stc_parcel_voxel_matrix.maximum(voxel_data).toarray()
    parcel_mins = stc_parcel_voxel_matrix.minimum(voxel_data).toarray()

    assert np.isfinite(parcel_means).all(), "NaN or inf values found in parcel means"
    assert np.isfinite(parcel_stds).all(), "NaN or inf values found in parcel standard deviations"
    assert np.isfinite(parcel_maxs).all(), "NaN or inf values found in parcel maximums"
    assert np.isfinite(parcel_mins).all(), "NaN or inf values found in parcel minimums"

    return np.stack([parcel_means, parcel_stds, parcel_maxs, parcel_mins], axis=-1), parcel_sizes


def calculate_adjacency_list(atlas_data: np.ndarray) -> list[tuple[int, int]]:
    adjacency_list = set()
    for i in range(atlas_data.shape[0]):
        for j in range(atlas_data.shape[1]):
            for k in range(atlas_data.shape[2]):
                current_parcel = atlas_data[i, j, k]
                if current_parcel == 0:  # Skip background
                    continue
                neighbors = [
                    (i - 1, j, k), (i + 1, j, k),
                    (i, j - 1, k), (i, j + 1, k),
                    (i, j, k - 1), (i, j, k + 1)
                ]
                for ni, nj, nk in neighbors:
                    if all(0 <= coord < dim for coord, dim in zip((ni, nj, nk), atlas_data.shape)):
                        neighbor_parcel = atlas_data[ni, nj, nk]
                        if neighbor_parcel != 0 and neighbor_parcel != current_parcel:
                            adjacency_list.add(tuple(sorted([current_parcel, neighbor_parcel])))
    return list(adjacency_list)


def get_atlas_data(stc, structurals_data):
    ho_atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')
    atlas_img = ho_atlas['maps']
    atlas_resampled = resample_atlas_to_stc(stc, structurals_data.src, atlas_img)
    return atlas_resampled.get_fdata().astype(int)


def create_voxel_to_atlas_mapping(structurals_data, atlas_data):
    pos = structurals_data.src[0]['rr'][structurals_data.src[0]['vertno']]
    return {
        i: tuple(np.round(p * 1000).astype(int))
        for i, p in enumerate(pos)
        if all(0 <= coord < dim for coord, dim in zip(np.round(p * 1000).astype(int), atlas_data.shape))
    }


def calculate_parcel_statistics(voxel_data, atlas_data, voxel_to_atlas_mapping):
    unique_parcels = np.unique(atlas_data)
    n_parcels = len(unique_parcels)
    n_times = voxel_data.shape[2]

    parcel_data = np.zeros((n_parcels, n_times, len(PARCEL_STATS), 3))
    parcel_sizes = np.zeros(n_parcels, dtype=int)

    for i, parcel in enumerate(unique_parcels):
        parcel_voxels = [
            voxel_data[voxel_idx]
            for voxel_idx, atlas_coords in voxel_to_atlas_mapping.items()
            if atlas_data[atlas_coords] == parcel
        ]

        if parcel_voxels:
            parcel_voxels = np.array(parcel_voxels)
            parcel_sizes[i] = len(parcel_voxels)

            for j, stat_func in enumerate(PARCEL_STATS):
                parcel_data[i, :, j, :] = stat_func(parcel_voxels, axis=0).T

    return parcel_data, parcel_sizes


def get_subject_parcels(path_args: PathArgs, read_subj_raw: callable, structurals_data: StructuralsData = None,
                        verbose: bool = False, cache: bool = False, recalculate: bool = False):
    stc = get_subject_voxels(path_args, read_subj_raw, structurals_data, verbose=verbose, cache=cache,
                             recalculate=recalculate)
    voxel_data = stc.data
    assert np.isfinite(voxel_data).all(), "NaN or inf values found in voxel data"

    atlas_data = get_atlas_data(stc, structurals_data)
    voxel_to_atlas_mapping = create_voxel_to_atlas_mapping(structurals_data, atlas_data)

    parcel_data, parcel_sizes = calculate_parcel_statistics(voxel_data, atlas_data, voxel_to_atlas_mapping)
    adjacency_list = calculate_adjacency_list(atlas_data)

    return VolParcelEstimate(parcel_data=parcel_data, parcel_sizes=parcel_sizes, adjacency_list=adjacency_list,
                             times=stc.times, sfreq=stc.sfreq)
