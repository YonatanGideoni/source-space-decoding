import matplotlib
import matplotlib.pyplot as plt

from consts import DEF_DATA_PATH

matplotlib.use('pdf')  # Use a non-interactive backend

import mne
import numpy as np

from data.data_utils import get_subject_raw
from data.dataset_utils import ArmeniSubloader
from data.path_utils import PathArgs

path_args = PathArgs(subj='001', sess='001', task='compr', data_path=DEF_DATA_PATH, dataset='armeni')

read_sraw = ArmeniSubloader.read_subj_raw
raw = get_subject_raw(path_args, read_sraw, cache=True)

# structurals = ArmeniSubloader.get_structurals(path_args)
# stc = get_subject_voxels(path_args, read_sraw, structurals, cache=True)

n_ts = len(raw.times)
labels = ArmeniSubloader.get_labels(path_args, sfreq=raw.info['sfreq'], n_ts=n_ts, labels='speech')

sfreq = raw.info['sfreq']  # Sampling frequency
times = np.arange(len(labels)) / sfreq  # Convert indices to time points
events = np.array([[int(round(t * sfreq)), 0, 1]
                   for i, t in enumerate(times[:-1])  # Iterate until the second last element
                   if labels[i] == 0 and labels[i + 1] == 1], dtype=int)

# Step 2: Create Epochs object around the events
event_id = 1  # The event id for the '1' label
tmin, tmax = -0.3, 0.3  # Time window around the event

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(-0.1, -0.05), preload=True)

# Step 3: Create Evoked object by averaging the epochs
evoked: mne.Evoked = epochs.average()

# Step 4: Plot the Evoked object
fs = 20
plt.rcParams.update({
    'font.size': fs,  # Global font size
    'axes.titlesize': fs + 4,  # Title font size
    'axes.labelsize': fs,  # Axes label font size
    'xtick.labelsize': fs,  # X-tick labels font size
    'ytick.labelsize': fs,  # Y-tick labels font size
    'legend.fontsize': fs,  # Legend font size
    'figure.titlesize': fs + 6  # Figure title font size
})

fig = evoked.plot_joint(times=[-0.2, 0., 0.05, 0.1, 0.15, 0.2, 0.25],
                        topomap_args={'contours': [-150, -100, -50, 0, 50, 100, 150]})
fig.set_size_inches(16, 8)

fig.savefig('plots/evoked.pdf')
