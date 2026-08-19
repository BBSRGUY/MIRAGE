import torch

from mirage.attention import spatiotemporal_mask


def test_sparse_mask_is_connected_and_sparse():
    mask = spatiotemporal_mask(4, 4, 4, 1, 4, torch.device("cpu"))
    assert mask.shape == (64, 64)
    assert mask.diagonal().all()
    assert 0 < mask.float().mean() < 1
