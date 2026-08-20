import torch

from mirage.experiments.sparse_residual import tile_sparse_residual


def test_tile_sparse_residual_selects_high_energy_tile():
    residual = torch.zeros(8, 8)
    residual[:4, :4] = 4
    residual[4:, 4:] = 1
    sparse, metrics = tile_sparse_residual(residual, 4, 0.25)
    assert torch.equal(sparse[:4, :4], residual[:4, :4])
    assert torch.count_nonzero(sparse[4:, 4:]) == 0
    assert metrics["selected_tiles"] == 1
    assert metrics["captured_residual_energy"] > 0.94
