import torch

from mirage.experiments.residual_spectrum import matrix_residual_spectrum


def test_residual_spectrum_identifies_exact_low_rank_matrix():
    torch.manual_seed(21)
    matrix = torch.randn(18, 3) @ torch.randn(3, 24)
    report = matrix_residual_spectrum(matrix, [1, 2, 3, 6], power_iterations=2, seed=21)
    assert report["ranks"]["3"]["captured_residual_energy"] > 0.99999
    assert report["ranks"]["3"]["remaining_relative_error"] < 0.005
