import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from consts import DEF_DATA_PATH
from data.data_aug_utils import create_all_dsets_vert2coords
from data.data_utils import get_subject_voxels, get_subject_raw
from data.dataset_utils import ArmeniSubloader
from data.path_utils import PathArgs
from train.parser_utils import get_base_parser


class MLP_VS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_VS, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


def learn_invert_sensor_from_voxels(stc_path: str, fif_path: str):
    # Read source estimate and raw data
    stc = mne.read_source_estimate(stc_path)
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    raw = raw.pick('mag')

    assert np.isclose(raw.times, stc.times).all()

    # Extract data from source estimate and raw data
    stc_data = stc.data.reshape(-1, stc.data.shape[-1]).T  # Shape: (time points, voxels)
    raw_data = raw.get_data().T  # Shape: (time points, sensors)

    # Ensure the data are aligned in time
    assert stc_data.shape[0] == raw_data.shape[0]

    # Train-test split
    stc_data = (stc_data - stc_data.mean(axis=0)) / stc_data.std(axis=0)
    raw_data = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(stc_data, raw_data, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the MLP model
    input_dim = X_train.shape[1]
    hidden_dim = 2048
    output_dim = y_train.shape[1]
    model = MLP_VS(input_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            print(f'Test Loss: {test_loss.item():.4f}')

    assert test_loss.item() < 1e-2, 'Error: source->sensor mapping is not invertible'


def plot_stc_psd(stc_path, fmin=0, fmax=100, nperseg=2048):
    stc = mne.read_source_estimate(stc_path)

    # Extract data and sampling frequency
    data = stc.data
    sfreq = stc.sfreq

    # Calculate PSD using Welch's method
    frequencies, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)

    # Find indices corresponding to fmin and fmax
    idx_min = np.argmax(frequencies >= fmin)
    idx_max = np.argmax(frequencies >= fmax) if fmax < frequencies[-1] else len(frequencies)

    # Plot the PSD
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies[idx_min:idx_max], psd.T[idx_min:idx_max])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.title('PSD of Source Time Courses')
    plt.xlim(fmin, fmax)
    plt.grid(True)
    plt.show()

    return frequencies[idx_min:idx_max], psd[:, idx_min:idx_max]


def create_v2cs():
    parser = get_base_parser()
    args = parser.parse_args()

    create_all_dsets_vert2coords(args)


def test_sensor_source_linear_invertibility(stc_path, fif_path):
    # Read source estimate and raw data
    stc = mne.read_source_estimate(stc_path)
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    raw.pick('mag')

    assert np.isclose(raw.times, stc.times).all()

    # Extract data from source estimate and raw data
    stc_data = stc.data.reshape(-1, stc.data.shape[-1]).T  # Shape: (time points, voxels)
    raw_data = raw.get_data().T  # Shape: (time points, sensors)

    # Ensure the data are aligned in time
    assert stc_data.shape[0] == raw_data.shape[0]

    # Train-test split
    stc_data = (stc_data - stc_data.mean(axis=0)) / stc_data.std(axis=0)
    raw_data = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(stc_data, raw_data, test_size=0.2, random_state=42)

    # fit an affine transformation
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)

    assert mse < 1e-10, 'Error: source->sensor mapping is not linearly invertible'


def main():
    path_args = PathArgs(subj='001', sess='001', task='compr', data_path=DEF_DATA_PATH, dataset='armeni')
    stc = get_subject_voxels(path_args, read_subj_raw=ArmeniSubloader.read_subj_raw,
                             structurals_data=ArmeniSubloader.get_structurals(path_args), verbose=True, cache=False)
    stc.save('temp', overwrite=True)

    raw = get_subject_raw(path_args, read_subj_raw=ArmeniSubloader.read_subj_raw, cache=True, verbose=True)

    raw.save('temp-raw.fif', overwrite=True)
    del stc, raw

    stc_path = 'temp-stc.h5'
    fif_path = 'temp-raw.fif'

    learn_invert_sensor_from_voxels(stc_path, fif_path)
    test_sensor_source_linear_invertibility(stc_path, fif_path)


if __name__ == '__main__':
    main()
