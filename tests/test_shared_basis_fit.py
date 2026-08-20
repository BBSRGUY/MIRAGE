import pytest
import torch

from mirage.compression.factorization import (
    FitOptions,
    compressed_parameter_count,
    fit_shared_basis,
)
from mirage.compression.streamed_fit import fit_streamed_shared_basis
from mirage.compression.hierarchical_fit import fit_hierarchical_shared_basis
from mirage.datasets import FeatureStore


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


def test_streamed_fit_matches_exact_shared_subspace(tmp_path):
    torch.manual_seed(9)
    basis = torch.randn(2, 7, 5)
    alpha = torch.randn(4, 2)
    weights = torch.einsum("lk,koi->loi", alpha, basis)
    store = FeatureStore(tmp_path)
    for index, weight in enumerate(weights):
        name = f"block.{index:02d}.attn.q"
        store.append(
            f"weight/{name}",
            {"weight": weight},
            kind="weight",
            sample_id="teacher",
            split="teacher",
            metadata={"name": name, "shape": list(weight.shape)},
        )
    records = list(store.records(kind="weight"))
    result = fit_streamed_shared_basis(
        store,
        records,
        (7, 5),
        FitOptions(basis_count=2, rank=0, seed=9),
        device=torch.device("cpu"),
        row_chunk_size=3,
    )
    assert torch.allclose(result.reconstruct(), weights, atol=2e-5, rtol=2e-5)
    assert result.metrics["solver"] == "streamed_layer_gram_pca"


def test_hierarchical_fit_reconstructs_group_subspaces(tmp_path):
    torch.manual_seed(12)
    global_basis = torch.randn(1, 6, 5)
    group_basis = torch.randn(2, 1, 6, 5)
    weights = []
    for index in range(6):
        group = index // 3
        weights.append((index + 1) * global_basis[0] + (index - 2) * group_basis[group, 0])
    weights = torch.stack(weights)
    store = FeatureStore(tmp_path)
    for index, weight in enumerate(weights):
        name = f"block.{index:02d}.attn.q"
        store.append(
            f"weight/{name}",
            {"weight": weight},
            kind="weight",
            sample_id="teacher",
            split="teacher",
            metadata={"name": name, "shape": list(weight.shape)},
        )
    result = fit_hierarchical_shared_basis(
        store,
        list(store.records(kind="weight")),
        (6, 5),
        groups=[list(range(3)), list(range(3, 6))],
        global_basis_count=1,
        group_basis_count=2,
        rank_per_layer=[0] * 6,
        device=torch.device("cpu"),
        row_chunk_size=2,
        seed=12,
    )
    reconstructed = torch.stack([result.reconstruct_layer(index) for index in range(6)])
    assert torch.allclose(reconstructed, weights, atol=5e-5, rtol=5e-5)
