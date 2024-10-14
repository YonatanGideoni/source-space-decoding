from itertools import product

import mne
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from consts import device
from data.data_aug_utils import RandomCubeZeroing, SliceDropout
from train.models import VoxelGCN, grid_to_voxel, voxel_to_grid, VoxelGNN
from train.parser_utils import get_base_parser
from train.training_utils import SpeechDetectionModel


def test_voxel_grid_inverse():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define test parameters
    batch_size = 2
    grid_len = 3
    grid_dims = (grid_len, grid_len, grid_len)
    n_grid_points = grid_len ** 3
    n_voxels = n_grid_points // 2  # Use half of the grid points as voxels
    n_channels = 3

    # Create random input data
    x = torch.rand(batch_size, n_voxels * n_channels)

    # Create vert_indices as a subset of grid points
    all_indices = torch.cartesian_prod(*[torch.arange(grid_len) for _ in range(3)])
    vert_indices = all_indices[torch.randperm(n_grid_points)[:n_voxels]]

    # Create a dummy data dictionary to match the voxel_to_grid function signature
    data = {'x': x}

    # Apply voxel_to_grid
    grid = voxel_to_grid(data, vert_indices, grid_dims)

    # Apply grid_to_voxel
    x_reconstructed = grid_to_voxel(grid, vert_indices, grid_dims)

    # Check if the reconstructed x is close to the original x
    is_close = torch.allclose(x, x_reconstructed, rtol=1e-5, atol=1e-5)

    if is_close:
        print("Test passed: voxel_to_grid and grid_to_voxel are inverses of each other.")
    else:
        print("Test failed: voxel_to_grid and grid_to_voxel are not inverses of each other.")
        print(f"Max difference: {(x - x_reconstructed).abs().max().item()}")
        print(f"Mean difference: {(x - x_reconstructed).abs().mean().item()}")
        raise AssertionError("voxel_to_grid and grid_to_voxel are not inverses of each other.")

    return is_close


def test_calculating_balanced_accuracy():
    # because of how pytorch lightning accumulates metrics balanced acc can be calculated dangeruosly incorrectly,
    # eg. when the batch size is 1
    parser = get_base_parser()
    args = parser.parse_args()

    sdm = SpeechDetectionModel(input_dim=1, args=args)
    sdm.eval()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.linear = torch.nn.Linear(1, 1)

        def forward(self, data):
            return torch.ones_like(data['x'])

    sdm.model = Model()
    trainer = Trainer(max_epochs=0)
    trainer.fit(sdm, train_dataloaders=[DataLoader([{'x': torch.tensor([1.0]), 'y': torch.tensor([1.0])}])])

    # generate some fake imbalanced data
    imbalance_ratio = 0.2
    n_samples = 1000
    n_positives = int(n_samples * imbalance_ratio)
    n_negatives = n_samples - n_positives
    y = torch.cat([torch.ones(n_positives), torch.zeros(n_negatives)])
    x = torch.randn(n_samples, 1)
    data = [{'x': x_, 'y': y_} for x_, y_ in zip(x, y)]

    # see that the model gives the correct balanced accuracy regardless of the dloader's batch size
    batch_sizes = [1, 2, 3, 4, 10, 100]
    for batch_size in batch_sizes:
        dloader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)

        acc = trainer.test(sdm, dataloaders=dloader)[0]

        acc = acc['test_acc']

        assert np.isclose(acc, .5), f'Balanced accuracy is {acc}, expected 0.5'


def test_random_cube_zeroing():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create sample 3D grids with different shapes, now with 6 channels (3 data + 3 positional)
    samples = [
        torch.ones((2, 6, 10, 10, 10)),
        torch.ones((3, 6, 20, 15, 25)),
        torch.ones((1, 6, 5, 5, 5)),
        torch.ones((5, 6, 30, 30, 30))
    ]

    transform = RandomCubeZeroing()

    for i, img in enumerate(samples):
        # Add positional embeddings
        img[:, -3:] = torch.randn_like(img[:, -3:])
        img = img.to(device)

        result = transform(img)

        # Check if the output has the same shape as the input
        assert result.shape == img.shape, f"Output shape doesn't match input shape for sample {i}"

        # Check if at least one voxel is zeroed out in each batch across data channels
        assert torch.any(result[:, :3] == 0, dim=(1, 2, 3, 4)).all(), \
            f"Some batches in sample {i} have no zeroed out voxels in data channels"

        # Check if the zeroed out regions are cubes (consistent across data channels)
        zero_mask = (result[:, :3] == 0).all(dim=1)  # Check if all data channels are zero
        for b in range(img.shape[0]):
            if zero_mask[b].any():
                nonzero_indices = torch.nonzero(zero_mask[b])
                mins = nonzero_indices.min(dim=0).values
                maxs = nonzero_indices.max(dim=0).values
                assert torch.all(zero_mask[b, mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1]), \
                    f"Zeroed out region in batch {b} of sample {i} is not a cube"

        # Check if different batch elements have different zero cubes
        if img.shape[0] > 1:  # Only perform this check if batch size > 1
            are_different = False
            for b1 in range(img.shape[0]):
                for b2 in range(b1 + 1, img.shape[0]):
                    if not torch.equal(zero_mask[b1], zero_mask[b2]):
                        are_different = True
                        break
                if are_different:
                    break
            assert are_different, f"All batch elements in sample {i} have identical zero cubes"

        # Check if all data channels are consistent in zeroing
        for b in range(img.shape[0]):
            for c in range(3):  # Only check data channels
                assert torch.all(zero_mask[b] == (result[b, c] == 0)), \
                    f"Zeroing is not consistent across data channels for batch {b} in sample {i}"




def test_slice_dropout():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create sample 3D grids with different shapes, now with 6 channels
    samples = [
        torch.ones((2, 6, 30, 30, 30)),
        torch.ones((3, 6, 20, 35, 25)),
        torch.ones((1, 6, 25, 25, 25)),
        torch.ones((5, 6, 40, 20, 45))
    ]

    dropout_prob = 0.2
    transform = SliceDropout(dropout_prob)

    for i, img in enumerate(samples):
        img = img.to(device=device)
        result = transform(img)

        # Check if the output has the same shape as the input
        assert result.shape == img.shape, f"Output shape doesn't match input shape for sample {i}"

        # Check if dropout has been applied (some values should be zero)
        assert torch.any(result == 0), f"No dropout applied in sample {i}"

        # Check if dropout is applied consistently across channels for each spatial location
        zero_mask = (result == 0)
        for b in range(img.shape[0]):
            assert torch.all(zero_mask[b, 0] == zero_mask[b, 1:]), \
                f"Dropout is not consistent across channels for batch {b} in sample {i}"

        # Check if dropout is applied along each axis
        for axis in [2, 3, 4]:  # x, y, z axes
            for b in range(img.shape[0]):
                slices_all_zero = torch.all(result[b] == 0, dim=tuple(set(range(4)) - {axis - 1}))
                assert torch.any(slices_all_zero), f"No full slice dropout along axis {axis} in sample {i}"

        # Check if the dropout rate is approximately correct
        dropout_rate = (result == 0).float().mean().item()
        exp_drpt_rate = 3 * dropout_prob
        assert 0.6 * exp_drpt_rate < dropout_rate < 1.4 * exp_drpt_rate, \
            f"Dropout rate {dropout_rate} is not close to the expected {exp_drpt_rate} for sample {i}"


def voxel_to_grid_wrapped(data):
    grid_dims = [4, 4, 4]
    vert_indices = torch.tensor([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
    ])
    return voxel_to_grid(data, vert_indices, grid_dims)


def test_v2g():
    x_single = torch.rand(2, 12)  # [batch_size, n_voxels * channels]
    data_single = {'x': x_single}
    result_single = voxel_to_grid_wrapped(data_single)
    exp_shape = (2, 6, 4, 4, 4)
    assert result_single.shape == exp_shape, f"Expected shape {exp_shape}, but got {result_single.shape}"


def test_value_placement():
    x_test = torch.zeros(1, 12, 1)
    x_test[0, 0::3, 0] = 1  # Set first channel to 1
    x_test[0, 1::3, 0] = 2  # Set second channel to 2
    x_test[0, 2::3, 0] = 3  # Set third channel to 3
    data_test = {'x': x_test}
    result_test = voxel_to_grid_wrapped(data_test)

    assert result_test[0, 0, 0, 0, 0] == 1, f"Expected 1, but got {result_test[0, 0, 0, 0, 0]}"
    assert result_test[0, 1, 0, 0, 0] == 2, f"Expected 2, but got {result_test[0, 1, 0, 0, 0]}"
    assert result_test[0, 2, 0, 0, 0] == 3, f"Expected 3, but got {result_test[0, 2, 0, 0, 0]}"
    assert result_test[0, 0, 1, 1, 1] == 1, f"Expected 1, but got {result_test[0, 0, 1, 1, 1]}"
    assert result_test[0, 1, 1, 1, 1] == 2, f"Expected 2, but got {result_test[0, 1, 1, 1, 1]}"
    assert result_test[0, 2, 1, 1, 1] == 3, f"Expected 3, but got {result_test[0, 2, 1, 1, 1]}"


def test_invalid_input_shape():
    x_invalid = torch.rand(2, 13, 1)  # 13 is not divisible by 3 channels
    data_invalid = {'x': x_invalid}
    try:
        voxel_to_grid_wrapped(data_invalid)
        assert False, "Expected AssertionError was not raised"
    except AssertionError:
        pass


def test_invalid_vert_indices():
    from train.models import voxel_to_grid

    grid_dims = [4, 4, 4]
    vert_indices = torch.tensor([[0, 0]])  # Invalid shape
    x_valid = torch.rand(2, 12, 1)
    data_valid = {'x': x_valid}
    try:
        voxel_to_grid(data_valid, vert_indices, grid_dims)
        assert False, "Expected AssertionError was not raised"
    except AssertionError:
        pass


class Args:
    def __init__(self, hidden_dim, dropout_prob):
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.vol_aug = None
        self.input_dropout_prob = 0.5
        self.dropout_prob = dropout_prob
        self.data_space = 'source'
        self.dataset = 'armeni'


def precompute_indices(args):
    # Example precomputed voxel indices for a 3x3x3 grid (27 voxels)
    grid_dims = (3, 3, 3)
    voxel_size = (1, 1, 1)
    voxel_indices = np.array(list(product(range(3), repeat=3)))  # 27 voxels, each with 3D coordinates
    return voxel_indices, grid_dims, voxel_size


def setup_gcn(batch_size=1):
    args = Args(hidden_dim=16, dropout_prob=0.2)
    VoxelGNN.precompute_indices = precompute_indices
    model = VoxelGCN(input_dim=None, args=args).to(device)

    # Mock data: a batch of 1 with 27 voxels, each having 3 features
    data = {
        'x': torch.randn(batch_size, 27 * 3).to(device)  # Shape: [batch_size, num_voxels * num_channels]
    }
    return model, data


def test_gcn_forward_pass():
    model, data = setup_gcn()
    output = model(data)

    # Check if the output shape is as expected: [batch_size, output_dim]
    assert output.shape == (1, 1), f"Expected output shape (1, 1), but got {output.shape}"
    print("Forward pass test passed.")


def test_gcn_graph_construction():
    model, data = setup_gcn()
    x_reshaped, edge_index = model.build_graph(data)

    # Check the reshaped input shape
    expected_shape = (1, 27, 6)  # 27 voxels with 3 features per voxel+3 features for pos embedding
    assert x_reshaped.shape == expected_shape, f"Expected node feature shape {expected_shape}, but got {x_reshaped.shape}"

    # Check the adjacency matrix shape (edge_index), should have edges for all connected neighbors
    # For a 3x3x3 grid, we expect each voxel to have up to 6 neighbors
    assert edge_index.shape[0] == 2, "edge_index should have two rows for source and target nodes."

    # Each voxel should be connected to its neighbors. Rough estimate: 27 voxels, each with ~6 connections = ~162 edges
    assert edge_index.shape[1] > 100, f"Expected at least 100 edges, but got {edge_index.shape[1]}"

    print("Graph construction test passed.")


def test_gcn_intermediate_dimensions():
    model, data = setup_gcn()
    x_reshaped, edge_index = model.build_graph(data)

    hidden_dim = model.gcn1.out_channels
    # Pass through the first GCN layer and check the output dimensions
    x_gcn1 = F.relu(model.gcn1(x_reshaped, edge_index))
    expected_shape = (1, 27, hidden_dim)
    assert x_gcn1.shape == expected_shape, f"Expected GCN1 output shape {expected_shape}, but got {x_gcn1.shape}"

    print("Intermediate node feature dimensions test passed.")


def test_gcn_batch_handling():
    model, data = setup_gcn(batch_size=4)

    # Forward pass with batch size of 4
    output = model(data)

    # Check that the model output matches the batch size
    assert output.shape == (4, 1), f"Expected output shape (4, 1) for batch size 4, but got {output.shape}"

    print("Batch handling test passed.")


def test_shared_source_forward_pass():
    stc = mne.read_source_estimate('smol_volest-stc.h5')
    x = stc.data
    x = torch.tensor(x).float().to(device).reshape(-1, x.shape[-1]).T

    models = ['MLP', 'GCN', 'SECNN']

    parser = get_base_parser()
    args = parser.parse_args()

    args.dataset = 'schoffelen'
    args.data_space = 'shared_source'
    args.multi_subj = True

    subject_inds = np.zeros(x.shape[0], dtype=int)
    subject_inds[0] = 1
    subject_inds[100] = 1
    subject_inds[200] = -1
    subject_inds = torch.tensor(subject_inds).to(device)

    for model_name in models:
        args.model = model_name

        data = {'x': x, 'subject_inds': subject_inds}
        model = SpeechDetectionModel(input_dim=x.shape[1], args=args).to(device)
        output = model(data)

        assert output.shape == (x.shape[0], 1), f"Expected output shape ({x.shape[0], 1}), but got {output.shape}"


def run_all_tests():
    test_functions = [
        test_random_cube_zeroing,
        test_shared_source_forward_pass,
        test_slice_dropout,
        test_voxel_grid_inverse,
        test_gcn_graph_construction,
        test_gcn_forward_pass,
        test_gcn_intermediate_dimensions,
        test_gcn_batch_handling,
        test_v2g,
        test_value_placement,
        test_invalid_input_shape,
        test_invalid_vert_indices,
        test_calculating_balanced_accuracy,
    ]

    for test_func in test_functions:
        try:
            test_func()
            print(f"{test_func.__name__} passed")
        except AssertionError as e:
            print(f"{test_func.__name__} failed: {str(e)}")

            raise e


if __name__ == '__main__':
    run_all_tests()
