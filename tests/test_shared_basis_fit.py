import pytest
import torch

from mirage.compression.factorization import (
    FitOptions,
    compressed_parameter_count,
    fit_shared_basis,
)


def test_exact_shared_subspace_reconstruction():
    torch.manual_seed(4)
    basis = torch.randn(2, 5, 4)
    alpha = torch.randn(6, 2)
    weights = torch.einsum("lk,koi->loi", alpha, basis)
    result = fit_shared_basis(weights, FitOptions(basis_count=2, rank=0, seed=4))
    assert torch.allclose(result.reconstruct(), weights, atol=2e-5, rtol=2e-5)
    assert result.metrics["weight_relative_error"] < 1e-5


def test_parameter_accounting_and_invalid_rank():
    assert compressed_parameter_count(6, 5, 4, 2, 1) == 2 * 5 * 4 + 6 * 2 + 6 * (5 + 4)
    with pytest.raises(ValueError, match="rank"):
        fit_shared_basis(torch.randn(3, 4, 5), FitOptions(basis_count=2, rank=5))
