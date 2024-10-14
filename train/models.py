from itertools import product

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mne_bids import BIDSPath
from scipy.spatial.distance import cdist
from monai.data import MetaTensor
from monai.transforms import RandGaussianNoise, RandGibbsNoise, RandBiasField
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SGConv
from torch_geometric.utils import to_dense_batch

import consts
from consts import device
from data.data_aug_utils import get_default_vert2coords, RandomCubeZeroing, SliceDropout
from data.path_utils import PathArgs


def precompute_indices(args):
    vert2coords = torch.tensor(get_default_vert2coords(args), dtype=torch.float32, device=consts.device)
    # Calculate voxel_size as the minimum non-zero difference in one dimension (e.g., x dimension)
    diffs_x = torch.unique(vert2coords[:, 0].sort().values.diff())
    voxel_size = diffs_x[diffs_x > 0].min().item()

    # Calculate the grid dimensions
    min_coords = vert2coords.min(dim=0)[0]
    max_coords = vert2coords.max(dim=0)[0]
    grid_dims = ((max_coords - min_coords) / voxel_size).round().long() + 1  # Ensure enough space

    # Shift coordinates to positive range and normalize by voxel size
    vert_indices = torch.round((vert2coords - min_coords) / voxel_size).long()

    return vert_indices, grid_dims, voxel_size


def voxel_to_grid(data, vert_indices, grid_dims) -> torch.Tensor:
    x = data['x']
    bs, n_voxels_channels = x.shape[0], x.shape[1]
    n_channels = 3  # Assuming 3 channels (x y z components) per voxel
    n_voxels = n_voxels_channels // n_channels

    # Assert that the input shape is correct
    assert n_voxels_channels % n_channels == 0, f"Input shape {x.shape} is not divisible by {n_channels} channels"

    # Assert that vert_indices has the correct shape
    assert vert_indices.shape[1] == 3, f"vert_indices should have shape (n_voxels, 3), but got {vert_indices.shape}"
    assert vert_indices.shape[0] == n_voxels, \
        f"vert_indices should have {n_voxels} rows, but got {vert_indices.shape[0]}"

    # Create a large zeros tensor of shape [bs, channels+3, grid_dims[0] * grid_dims[1] * grid_dims[2], n_ts]
    grid_flat = torch.zeros((bs, n_channels + 3, grid_dims[0] * grid_dims[1] * grid_dims[2]), dtype=x.dtype,
                            device=x.device)

    # Calculate the linear index for each voxel
    linear_indices = (vert_indices[:, 0] * (grid_dims[1] * grid_dims[2]) +
                      vert_indices[:, 1] * grid_dims[2] +
                      vert_indices[:, 2])

    # Assert that linear_indices are within the expected range
    assert linear_indices.min() >= 0 and linear_indices.max() < grid_dims[0] * grid_dims[1] * grid_dims[2], \
        (f"linear_indices out of range: "
         f"min={linear_indices.min()}, "
         f"max={linear_indices.max()}, "
         f"expected max={grid_dims[0] * grid_dims[1] * grid_dims[2] - 1}")

    # Expand dimensions of linear_indices for broadcasting
    linear_indices_expanded = linear_indices.unsqueeze(0).expand(bs, -1)

    # Reshape x to separate channels: [bs, n_voxels, channels]
    x_reshaped = x.view(bs, n_voxels, n_channels)

    # Use scatter_ to place the voxel values into the flat grid for all channels and time steps at once
    for c in range(n_channels):
        grid_flat[:, c].scatter_(1, linear_indices_expanded, x_reshaped[:, :, c])

    # Normalize and add positional embeddings
    normalized_indices = 2 * (vert_indices.float() / torch.tensor(grid_dims).float().unsqueeze(0)) - 1
    for c in range(3):
        grid_flat[:, n_channels + c].scatter_(1, linear_indices_expanded,
                                              normalized_indices[:, c].unsqueeze(0).expand(bs, -1))

    # Reshape the flat grid back to the 3D grid with channels and time
    grid = grid_flat.view(bs, n_channels + 3, grid_dims[0], grid_dims[1], grid_dims[2])

    return grid


def grid_to_voxel(grid, vert_indices, grid_dims):
    bs, n_channels_plus_3, *_ = grid.shape
    n_channels = n_channels_plus_3 - 3  # Subtract 3 for positional embeddings

    # Assert that the input shapes are correct
    assert grid.shape[2:] == tuple(grid_dims), f"Grid shape {grid.shape[2:]} doesn't match grid_dims {grid_dims}"
    assert vert_indices.shape[1] == 3, f"vert_indices should have shape (n_voxels, 3), but got {vert_indices.shape}"

    # Flatten the grid
    grid_flat = grid.view(bs, n_channels_plus_3, -1)

    # Calculate the linear index for each voxel
    linear_indices = (vert_indices[:, 0] * (grid_dims[1] * grid_dims[2]) +
                      vert_indices[:, 1] * grid_dims[2] +
                      vert_indices[:, 2])

    # Assert that linear_indices are within the expected range
    assert linear_indices.min() >= 0 and linear_indices.max() < grid_dims[0] * grid_dims[1] * grid_dims[2], \
        (f"linear_indices out of range: "
         f"min={linear_indices.min()}, "
         f"max={linear_indices.max()}, "
         f"expected max={grid_dims[0] * grid_dims[1] * grid_dims[2] - 1}")

    # Expand dimensions of linear_indices for broadcasting
    linear_indices_expanded = linear_indices.unsqueeze(0).unsqueeze(1).expand(bs, n_channels, -1)

    # Extract voxel values from the flat grid
    x = torch.gather(grid_flat[:, :n_channels], 2, linear_indices_expanded)

    # Reshape x to the original flattened format: [bs, n_voxels * n_channels]
    x = x.transpose(1, 2).contiguous().view(bs, -1)

    return x


class MLP(nn.Module):
    def __init__(self, input_dim, args, n_subjects: int = None, embed_dim: int = 16):
        super(MLP, self).__init__()
        self.subject_embeddings = None
        multisubj = n_subjects is not None and n_subjects > 1
        if multisubj:
            self.subject_embeddings = nn.Embedding(num_embeddings=n_subjects, embedding_dim=embed_dim)
            input_dim += embed_dim

        hidden_dim = args.hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + embed_dim * multisubj, hidden_dim)
        self.extra_layer = args.extra_layer
        if self.extra_layer:
            self.fc3 = nn.Linear(hidden_dim + embed_dim * multisubj, hidden_dim)

        self.output = nn.Linear(hidden_dim + embed_dim * multisubj, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout_prob)

        self.input_dropout = nn.Dropout(args.input_dropout_prob)

        if 'source' in args.data_space:
            self.slice_drpt = SliceDropout(args.slice_dropout_prob)
            self.cbdrpt = RandomCubeZeroing(args.cb_dropout_prob)

            self.vert_indices, self.grid_dims, self.voxel_size = precompute_indices(args)

    def get_subj_embed(self, subject_inds):
        # Compute the average embedding across all subjects
        avg_embedding = self.subject_embeddings.weight.mean(dim=0)

        # Mask to identify which subjects are unseen (-1)
        unseen_mask = (subject_inds == -1)

        # Handle seen subjects: Get embeddings for valid indices (i.e., subject_inds >= 0)
        seen_subject_inds = subject_inds[~unseen_mask]  # Only valid (non-negative) subject indices
        if seen_subject_inds.numel() > 0:  # If there are any seen subjects
            seen_embeddings = self.subject_embeddings(seen_subject_inds)

        # Prepare the full embedding tensor, initially filled with the average embedding for all
        subj_embeddings = avg_embedding.expand(subject_inds.size(0), -1).clone()

        # If there are seen subjects, place their embeddings in the corresponding positions
        if seen_subject_inds.numel() > 0:
            subj_embeddings[~unseen_mask.squeeze()] = seen_embeddings

        return subj_embeddings.squeeze(1)

    def forward(self, data: dict):
        x, subject_inds = data['x'], data['subject_inds']
        x = x.squeeze(-1)

        x = self.input_dropout(x)  # don't dropout subject embeddings

        if self.training and hasattr(self, 'slice_drpt') and (self.slice_drpt.prob > 0 or self.cbdrpt.prob > 0):
            grid = voxel_to_grid(data, self.vert_indices, self.grid_dims)

            grid = self.slice_drpt(grid)
            grid = self.cbdrpt(grid)

            x = grid_to_voxel(grid, self.vert_indices, self.grid_dims)

        has_subj_embed = self.subject_embeddings is not None
        if has_subj_embed:
            subj_embeddings = self.get_subj_embed(subject_inds)
            assert x.ndim == subj_embeddings.ndim, f'ndims do not match: {x.shape=} {subj_embeddings.shape=}'
            x = torch.cat((x, subj_embeddings), dim=-1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x1 = x

        x = torch.cat((x, subj_embeddings), dim=-1) if has_subj_embed else x
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        if self.extra_layer:
            x = torch.cat((x, subj_embeddings), dim=-1) if has_subj_embed else x
            x = self.fc3(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = x + x1

        x = torch.cat((x, subj_embeddings), dim=-1) if has_subj_embed else x
        x = self.output(x)

        return x


class VolumetricCNN(nn.Module):
    def __init__(self, input_dim, args, n_subjects: int = None, aug_prob: float = 0.5, **kwargs):
        super(VolumetricCNN, self).__init__()

        # Precompute indices and grid dimensions
        self.vert_indices, self.grid_dims, self.voxel_size = precompute_indices(args)

        self.subject_embeddings = None
        if n_subjects is not None and n_subjects > 1:
            self.subject_embeddings = nn.Embedding(num_embeddings=n_subjects, embedding_dim=input_dim)

        volumentation_dict = {
            'RandomBiasField': RandBiasField(prob=aug_prob),
            'RandomGaussianNoise': RandGaussianNoise(prob=aug_prob),
            'RandomGibbsNoise': RandGibbsNoise(prob=aug_prob),
            'CubeZeroing': RandomCubeZeroing(prob=aug_prob),
            'SliceDropout': SliceDropout(prob=args.slice_dropout_prob)
        }

        # Get the selected augmentation
        assert args.slice_dropout_prob is None or args.vol_aug is None, \
            "Cannot use both slice dropout and another augmentation yet"
        vol_aug = args.vol_aug if args.vol_aug != 0 else 'SliceDropout' if args.slice_dropout_prob > 0 else None
        assert vol_aug is None or args.cb_dropout_prob == 0., "Cannot use both CubeZeroing and another augmentation"
        vol_aug = 'CubeZeroing' if args.cb_dropout_prob > 0. else vol_aug
        self.augmentation = volumentation_dict.get(vol_aug, None)

        self.input_dropout = nn.Dropout3d(args.input_dropout_prob)

        hidden_dim = args.hidden_dim
        self.conv1 = nn.Conv3d(in_channels=6, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.dropout_3d = nn.Dropout3d(args.dropout_prob)
        # division due to max pooling
        self.fc1 = nn.Linear(hidden_dim *
                             (self.grid_dims[0] // 2) *
                             (self.grid_dims[1] // 2) *
                             (self.grid_dims[2] // 2), hidden_dim)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        # Add subject embeddings if applicable
        if self.subject_embeddings is not None:
            subj_inds = data['subject_inds']
            subj_embeddings = self.subject_embeddings(subj_inds.clamp(min=0)).swapaxes(1, 2)

            unk_mask = subj_inds == -1
            if unk_mask.any():
                # Calculate the average embedding
                avg_embedding = self.subject_embeddings.weight.mean(dim=0, keepdim=True).T

                # Create a mask for -1 indices
                unk_mask = unk_mask.unsqueeze(1).expand_as(subj_embeddings)

                # Replace embeddings for -1 indices with the average embedding
                subj_embeddings = torch.where(unk_mask, avg_embedding, subj_embeddings)

            data['x'] += subj_embeddings

        # Convert voxel data to 3D grid
        grid = voxel_to_grid(data, self.vert_indices, self.grid_dims)

        # Apply augmentation if specified
        if self.augmentation is not None and self.training:
            grid = MetaTensor(grid)  # Convert to MetaTensor for MONAI
            grid = self.augmentation(grid)
            grid = grid.as_tensor()  # Convert back to regular tensor

        out = torch.relu(self.conv1(grid))
        out = self.dropout_3d(out)

        out = torch.relu(self.conv2(out))
        out = self.dropout_3d(out)
        out = self.pool(out)

        # Flatten the output for the fully connected layer
        out = out.reshape(out.shape[0], -1)

        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SEVolumetricCNN(VolumetricCNN):
    def __init__(self, input_dim, args, n_subjects: int = None, aug_prob: float = 0.5, **kwargs):
        super(SEVolumetricCNN, self).__init__(input_dim, args, n_subjects, aug_prob, **kwargs)

        hidden_dim = args.hidden_dim
        self.se1 = SEBlock(channel=hidden_dim)
        self.se2 = SEBlock(channel=hidden_dim)

        self.extra_layer = args.extra_layer
        if self.extra_layer:
            self.conv3 = nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.se3 = SEBlock(channel=hidden_dim)

    def forward(self, data):
        # Add subject embeddings if applicable
        if self.subject_embeddings is not None:
            subj_inds = data['subject_inds']
            subj_embeddings = self.subject_embeddings(subj_inds.clamp(min=0)).swapaxes(1, 2)

            unk_mask = subj_inds == -1
            if unk_mask.any():
                # Calculate the average embedding
                avg_embedding = self.subject_embeddings.weight.mean(dim=0, keepdim=True).T

                # Create a mask for -1 indices
                unk_mask = unk_mask.unsqueeze(1).expand_as(subj_embeddings)

                # Replace embeddings for -1 indices with the average embedding
                subj_embeddings = torch.where(unk_mask, avg_embedding, subj_embeddings)

            data['x'] += subj_embeddings

        # Convert voxel data to 3D grid
        grid = voxel_to_grid(data, self.vert_indices, self.grid_dims)

        # Apply augmentation if specified
        if self.augmentation is not None and self.training:
            grid = MetaTensor(grid)  # Convert to MetaTensor for MONAI
            grid = self.augmentation(grid)
            grid = grid.as_tensor()  # Convert back to regular tensor

        # Pass the grid through the CNN layers with SE blocks
        out = torch.relu(self.conv1(grid))
        out = self.dropout_3d(out)
        out = self.se1(out)
        out1 = out

        out = torch.relu(self.conv2(out))
        out = self.dropout_3d(out)
        out = self.se2(out)

        if self.extra_layer:
            out = torch.relu(self.conv3(out))
            out = self.dropout_3d(out)
            out = self.se3(out)
            out = out + out1

        out = self.pool(out)

        # Flatten the output for the fully connected layer
        out = out.reshape(out.shape[0], -1)

        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class VoxelGNN(nn.Module):
    precompute_indices = precompute_indices

    def __init__(self, input_dim, args, n_subjects: int = None, precompute_inds: bool = True):
        super(VoxelGNN, self).__init__()

        if precompute_inds:
            self.vert_indices, self.grid_dims, _ = VoxelGNN.precompute_indices(args)
            if isinstance(self.vert_indices, torch.Tensor):
                self.vert_indices = self.vert_indices.detach().cpu().numpy()
            self.vert_indices = [tuple(vert) for vert in self.vert_indices]

            self.edge_index = self.create_adjacency_matrix(self.vert_indices)
            # Normalize vertex indices to [-1, 1]
            self.normalized_indices = self.normalize_indices(self.vert_indices)

            self.n_voxels = len(self.vert_indices)

            self.in_channels = 6  # xyz vector components+3 positional embeddings

        # Subject embedding layer
        self.n_subjects = n_subjects
        self.subject_embeddings = None
        if n_subjects is not None:
            self.subject_embeddings = nn.Embedding(n_subjects, input_dim)

    def normalize_indices(self, indices):
        indices = np.array(indices)
        min_vals = indices.min(axis=0)
        max_vals = indices.max(axis=0)
        normalized = 2 * (indices - min_vals) / (max_vals - min_vals) - 1
        return torch.tensor(normalized, dtype=torch.float32)

    def build_graph(self, data):
        """
        Converts voxel data into a graph with voxel-to-voxel connections.
        The voxel graph will have nodes as voxel intensities and edges as neighbors.
        """
        x = data['x']  # Node features (voxel data)
        bs, n_voxels_channels = x.shape[0], x.shape[1]
        n_channels = 3  # 3 channels per voxel, vector components (x, y, z)

        # Reshape x to [batch_size, n_voxels, 3] - separating the voxel features
        n_voxels = n_voxels_channels // n_channels
        x_reshaped = x.view(bs, n_voxels, n_channels)

        # Add positional embeddings
        pos_embeddings = self.normalized_indices.unsqueeze(0).repeat(bs, 1, 1).to(x_reshaped.device)
        x_with_pos = torch.cat([x_reshaped, pos_embeddings], dim=-1)

        return x_with_pos, self.edge_index

    def create_adjacency_matrix(self, voxel_indices: list[tuple]):
        edge_list = []
        for i, (x, y, z) in enumerate(voxel_indices):
            neighbors = [(x + dx, y + dy, z + dz) for dx, dy, dz in product([-1, 0, 1], repeat=3) if
                         abs(dx) + abs(dy) + abs(dz) == 1]

            for neighbor in neighbors:
                if neighbor in voxel_indices:
                    try:
                        j = voxel_indices.index(neighbor)
                        edge_list.append([i, j])
                    except ValueError:
                        continue

        edge_index = torch.tensor(edge_list).t().contiguous()
        return edge_index.to(device)

    def get_subject_embeddings(self, data):
        subj_inds = data['subject_inds']
        subj_embeddings = self.subject_embeddings(subj_inds.clamp(min=0)).swapaxes(1, 2)

        unk_mask = subj_inds == -1
        if unk_mask.any():
            # Calculate the average embedding
            avg_embedding = self.subject_embeddings.weight.mean(dim=0, keepdim=True).T

            # Create a mask for -1 indices
            unk_mask = unk_mask.unsqueeze(1).expand_as(subj_embeddings)

            # Replace embeddings for -1 indices with the average embedding
            subj_embeddings = torch.where(unk_mask, avg_embedding, subj_embeddings)

        return subj_embeddings.to(device)


class VoxelGCN(VoxelGNN):
    def __init__(self, input_dim, args, n_subjects: int = None):
        super(VoxelGCN, self).__init__(input_dim, args, n_subjects)

        hidden_dim = args.hidden_dim

        # Graph Convolutional layers
        self.gcn1 = GCNConv(self.in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * self.n_voxels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Dropout layers
        self.channel_dropout = nn.Dropout2d(args.dropout_prob)
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, data):
        if self.subject_embeddings is not None:
            data['x'] += self.get_subject_embeddings(data)

        x, edge_index = self.build_graph(data)

        x = F.relu(self.gcn1(x, edge_index))
        x = self.channel_dropout(x.swapaxes(1, 2).unsqueeze(-1)).squeeze(-1).swapaxes(1, 2)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.channel_dropout(x.swapaxes(1, 2).unsqueeze(-1)).squeeze(-1).swapaxes(1, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class VoxelGAT(VoxelGNN):
    def __init__(self, input_dim, args, n_subjects: int = None, precompute_inds: bool = True):
        super(VoxelGAT, self).__init__(input_dim, args, n_subjects, precompute_inds)

        hidden_dim = args.hidden_dim
        num_heads = 4

        # Graph Attention Network layers (multi-head attention)
        self.gat1 = GATConv(self.in_channels, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * num_heads * self.n_voxels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(args.dropout_prob)
        self.channel_dropout = nn.Dropout1d(args.dropout_prob)

    def forward(self, data):
        if self.subject_embeddings is not None:
            data['x'] += self.get_subject_embeddings(data)

        bs = data['x'].shape[0]
        x, edge_index = self.build_graph(data)
        x = [Data(x=x, edge_index=edge_index) for x in x]
        batched = Batch.from_data_list(x)

        # Apply GAT layers with channelwise dropout
        x = F.relu(self.gat1(batched.x, batched.edge_index))
        x = self.channel_dropout(x.view(bs, -1, x.size(-1)).swapaxes(2, 1)).swapaxes(2, 1).view(x.size())
        x = F.relu(self.gat2(x, batched.edge_index))
        x = self.channel_dropout(x.view(bs, -1, x.size(-1)).swapaxes(2, 1)).swapaxes(2, 1).view(x.size())

        x = to_dense_batch(x, batched['batch'])[0].reshape(bs, -1)

        # Apply fully connected layers with regular dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class VoxelSGC(VoxelGNN):
    def __init__(self, input_dim, args, n_subjects: int = None):
        super(VoxelSGC, self).__init__(input_dim, args, n_subjects)

        hidden_dim = args.hidden_dim

        # Simplified Graph Convolution layers
        self.sgc1 = SGConv(self.in_channels, hidden_dim, K=2)
        self.sgc2 = SGConv(hidden_dim, hidden_dim, K=2)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * self.n_voxels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        if self.subject_embeddings is not None:
            data['x'] += self.get_subject_embeddings(data)

        x, edge_index = self.build_graph(data)  # Node features, adjacency matrix

        x = F.relu(self.sgc1(x, edge_index))
        x = F.relu(self.sgc2(x, edge_index))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class SensorGAT(VoxelGAT):
    def __init__(self, input_dim, args, n_subjects: int = None):
        self.n_voxels = 273  # hardcoded hack for # of sensors in Schoffelen
        self.in_channels = 4  # 1 for sensor data, 3 for positional embeddings
        super(SensorGAT, self).__init__(input_dim, args, n_subjects, precompute_inds=False)

        assert args.data_space == 'sensor', 'SensorGAT is only implemented for sensor data'

        self.edge_index, self.positions = self.get_adj_mat_from_data(args)
        self.register_buffer('normalized_positions', torch.tensor(self.positions, dtype=torch.float).to(device))

    def precompute_indices(self, args):
        return None, None, None

    @classmethod
    def get_adj_mat_from_data(cls, args, n_neighbors: int = 5):
        # hack to get sensor positions
        assert args.dataset == 'schoffelen', 'SensorGAT is only implemented for Schoffelen'
        path_args = PathArgs(subj='A2002', sess=None, task='auditory', data_path=args.data_path, dataset='schoffelen')
        bids_path = BIDSPath(subject=path_args.subj, session=path_args.sess, task=path_args.task, root=path_args.root)
        raw = mne.io.read_raw_ctf(bids_path.fpath, preload=False, verbose=False)
        raw.pick(picks='mag')

        positions = np.array([ch['loc'][:3] for ch in raw.info['chs'] if ch['kind'] == 1])

        # Scale positions to [-1, 1] in each dimension
        positions = 2 * (positions - positions.min(axis=0)) / (positions.max(axis=0) - positions.min(axis=0)) - 1

        # Calculate the distance matrix
        dist_matrix = cdist(positions, positions)

        nearest_neighbors = np.argsort(dist_matrix, axis=1)[:, 1:n_neighbors + 1]

        # Create edge list
        rows = np.repeat(np.arange(len(positions)), n_neighbors)
        cols = nearest_neighbors.flatten()

        # Combine rows and cols to create edge_index
        edge_index = np.vstack((rows, cols))

        # Make the graph undirected by adding reverse edges
        edge_index = np.hstack((edge_index, edge_index[::-1]))

        # Remove duplicate edges
        edge_index = np.unique(edge_index, axis=1)

        # Convert to PyTorch tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

        return edge_index, positions

    def build_graph(self, data):
        """
        Converts sensor data into a graph with sensor-to-sensor connections.
        The sensor graph will have nodes as sensor readings and edges as neighbors.
        """
        x = data['x']  # Node features (sensor data)
        bs, n_sensors = x.shape[0], x.shape[1]

        # Add positional embeddings
        pos_embeddings = self.normalized_positions.unsqueeze(0).repeat(bs, 1, 1).to(x.device)
        x_with_pos = torch.cat([x, pos_embeddings], dim=-1)

        return x_with_pos, self.edge_index


def get_model(args) -> nn.Module:
    return {'MLP': MLP, 'CNN': VolumetricCNN, 'SECNN': SEVolumetricCNN,
            'GCN': VoxelGCN, 'GAT': VoxelGAT, 'SGC': VoxelSGC, 'SensorGAT': SensorGAT}[args.model]
