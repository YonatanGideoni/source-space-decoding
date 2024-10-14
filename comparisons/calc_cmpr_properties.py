import pickle

import mne
import numpy as np
from scipy.stats import pearsonr

from data.data_utils import get_subject_voxels, get_subject_raw
from data.dataset_utils import ArmeniSubloader
from train.parser_utils import get_base_parser
from data.path_utils import PathArgs

BASELINE = (-0.1, -0.05)


def extract_segments(stc: mne.SourceEstimate, indices: list[int],
                     sfreq: float, pre_stimulus: float = 0.3, post_stimulus: float = 0.3) -> list[mne.SourceEstimate]:
    """Extract segments from the source estimate data and set stimuli at time zero."""
    segments = []
    pre_samples = int(pre_stimulus * sfreq)
    post_samples = int(post_stimulus * sfreq)

    for idx in indices:
        start = idx - pre_samples
        end = idx + post_samples
        if start >= 0 and end < stc.data.shape[1]:
            segment_data = stc.data[:, start:end].copy()
            # Set the first time point to be -pre_stimulus, ensuring stimuli is at time zero
            tmin = -pre_stimulus
            # Create the new segment with adjusted time axis
            segment = mne.VolSourceEstimate(subject=stc.subject, tmin=tmin, tstep=stc.tstep,
                                            vertices=stc.vertices, data=segment_data, verbose=False)
            segments.append(segment)
    return segments


def average_segments(segments: list[mne.SourceEstimate], baseline=None) -> mne.SourceEstimate:
    """Average the extracted segments with optional baseline correction."""

    if baseline is not None:
        baseline_start, baseline_end = baseline
        # Convert baseline time to sample indices
        baseline_samples = segments[0].time_as_index([baseline_start, baseline_end])

        # Baseline correction for each segment
        for seg in segments:
            baseline_mean = np.mean(seg.data[:, baseline_samples[0]:baseline_samples[1]], axis=1, keepdims=True)
            # baseline_std = np.std(seg.data[:, baseline_samples[0]:baseline_samples[1]], axis=1, keepdims=True)
            seg.data -= baseline_mean
            # seg.data /= baseline_std

    # Average the segments
    avg_data = np.mean([seg.data for seg in segments], axis=0)
    avg_stc = segments[0].copy()  # Use the first segment as a template
    avg_stc.data = avg_data

    return avg_stc


def cache_epoched_data(path_args: PathArgs):
    read_sraw = ArmeniSubloader.read_subj_raw
    raw = get_subject_raw(path_args, read_subj_raw=read_sraw, cache=True, verbose=False)

    structurals = ArmeniSubloader.get_structurals(path_args)
    stc = get_subject_voxels(path_args, read_subj_raw=read_sraw, structurals_data=structurals, cache=True,
                             verbose=False)

    labels = ArmeniSubloader.get_labels(path_args, sfreq=raw.info['sfreq'], labels='speech', n_ts=stc.shape[1])

    # epoch data based on speech onset
    times = raw.times
    sfreq = raw.info['sfreq']
    events = np.array([[int(round(t * sfreq)), 0, 1] for i, t in enumerate(times[:-1])
                       if labels[i] == 0 and labels[i + 1] == 1], dtype=int)
    print(f'Found {len(events)} events')
    event_id = 1  # The event id for the '1' label
    tmin, tmax = -0.3, 0.3  # Time window around the event

    raw_epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=BASELINE, preload=True)
    stc_segs = extract_segments(stc, events[:, 0], sfreq)

    raw_evoked = raw_epochs.average()
    stc_evoked = average_segments(stc_segs, baseline=BASELINE)

    raw_evoked.save('raw_evoked-epo.fif', overwrite=True)
    stc_evoked.save('stc_evoked-epo', overwrite=True)


def cache_corrcoefs(path_args: PathArgs):
    read_sraw = ArmeniSubloader.read_subj_raw
    raw = get_subject_raw(path_args, read_subj_raw=read_sraw, cache=True, verbose=False)

    structurals = ArmeniSubloader.get_structurals(path_args)
    stc = get_subject_voxels(path_args, read_subj_raw=read_sraw, structurals_data=structurals, cache=True,
                             verbose=False)

    labels = ArmeniSubloader.get_labels(path_args, sfreq=raw.info['sfreq'], labels='speech', n_ts=stc.shape[1])

    raw_corrcoef = pearsonr(raw.get_data().T, labels.reshape(-1, 1)).statistic
    stc_corrcoef = pearsonr(stc.data.T, labels.reshape(-1, 1)).statistic

    assert not np.isnan(raw_corrcoef).any()
    assert not np.isnan(stc_corrcoef).any()

    with open('raw_corrcoef.pkl', 'wb') as f:
        pickle.dump(raw_corrcoef, f)

    with open('stc_corrcoef.pkl', 'wb') as f:
        pickle.dump(stc_corrcoef, f)


def main():
    parser = get_base_parser()
    args = parser.parse_args()

    path_args = PathArgs(subj='001', sess='001', task='compr', dataset='armeni', data_path=args.data_path)
    cache_corrcoefs(path_args)
    cache_epoched_data(path_args)


if __name__ == '__main__':
    main()
