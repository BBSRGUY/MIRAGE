import torch

from mirage.experiments.delta_spectrum import delta_spectrum


def test_delta_spectrum_detects_low_rank_change():
    torch.manual_seed(7)
    left = torch.randn(12, 3)
    right = torch.randn(3, 20)
    report = delta_spectrum(left @ right, [1, 2, 3, 8])
    assert report["ranks"]["3"]["explained_energy"] > 0.99999
    assert report["ranks"]["3"]["relative_reconstruction_error"] < 0.005
