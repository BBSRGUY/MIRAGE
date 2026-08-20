from __future__ import annotations

import math
from dataclasses import asdict
from time import perf_counter
from typing import Sequence

import torch
from safetensors import safe_open

from ..datasets import FeatureRecord, FeatureStore
from .factorization import FactorizationResult, FitOptions


def _weight_rows(
    store: FeatureStore,
    record: FeatureRecord,
    start: int,
    stop: int,
    device: torch.device,
) -> torch.Tensor:
    path = store.root / record.relative_path
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        rows = handle.get_slice("weight")[start:stop]
    return rows.to(device=device, dtype=torch.float32)


def streamed_layer_gram(
    store: FeatureStore,
    records: Sequence[FeatureRecord],
    shape: tuple[int, int],
    *,
    device: torch.device,
    row_chunk_size: int,
) -> torch.Tensor:
    """Compute W_flat @ W_flat.T without materializing the stacked matrices."""
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive")
    gram = torch.zeros(len(records), len(records), device=device, dtype=torch.float64)
    for start in range(0, shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, shape[0])
        chunk = torch.stack(
            [_weight_rows(store, record, start, stop, device) for record in records]
        ).flatten(1)
        gram.addmm_(chunk.double(), chunk.double().T)
        del chunk
    return gram.float()


def streamed_pca_basis(
    store: FeatureStore,
    records: Sequence[FeatureRecord],
    shape: tuple[int, int],
    *,
    basis_count: int,
    device: torch.device,
    row_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return PCA bases, exact coefficients, and layer-space eigenvalues."""
    gram = streamed_layer_gram(
        store, records, shape, device=device, row_chunk_size=row_chunk_size
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    usable = min(basis_count, int((eigenvalues > 1e-8 * eigenvalues.max()).sum().item()))
    basis = torch.zeros(
        basis_count, shape[0], shape[1], device="cpu", dtype=torch.float32
    )
    if usable:
        mixing = eigenvectors[:, :usable].T / eigenvalues[:usable].sqrt()[:, None].clamp_min(1e-12)
        for start in range(0, shape[0], row_chunk_size):
            stop = min(start + row_chunk_size, shape[0])
            chunk = torch.stack(
                [_weight_rows(store, record, start, stop, device) for record in records]
            )
            basis[:usable, start:stop].copy_(
                torch.einsum("kl,loi->koi", mixing, chunk).cpu()
            )
            del chunk
    alpha = torch.zeros(len(records), basis_count, dtype=torch.float32)
    if usable:
        alpha[:, :usable] = (
            eigenvectors[:, :usable] * eigenvalues[:usable].sqrt()[None]
        ).cpu()
    return basis, alpha, eigenvalues.cpu()


def _layer_low_rank(
    residual: torch.Tensor, rank: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if rank == 0:
        return residual.new_empty((residual.shape[0], 0)), residual.new_empty(
            (0, residual.shape[1])
        )
    effective = min(rank, min(residual.shape))
    with torch.random.fork_rng(devices=[residual.device] if residual.is_cuda else []):
        torch.manual_seed(seed)
        u, singular, v = torch.svd_lowrank(residual, q=effective, niter=2)
    root = singular.clamp_min(0).sqrt()
    return u * root[None], root[:, None] * v.T


def fit_streamed_shared_basis(
    store: FeatureStore,
    records: Sequence[FeatureRecord],
    shape: tuple[int, int],
    options: FitOptions,
    *,
    device: torch.device,
    row_chunk_size: int = 32,
    precomputed_basis: torch.Tensor | None = None,
    precomputed_alpha: torch.Tensor | None = None,
    eigenvalues: torch.Tensor | None = None,
) -> FactorizationResult:
    """Fit bases and layer residuals with bounded memory, one teacher matrix at a time."""
    if options.optimization_steps:
        raise ValueError("streamed fitting does not support joint dense optimization")
    if options.initialization != "pca":
        raise ValueError("streamed fitting currently requires PCA initialization")
    if options.basis_count > len(records):
        raise ValueError("basis_count exceeds layer count")
    start_time = perf_counter()
    if precomputed_basis is None or precomputed_alpha is None or eigenvalues is None:
        basis, alpha, eigenvalues = streamed_pca_basis(
            store,
            records,
            shape,
            basis_count=options.basis_count,
            device=device,
            row_chunk_size=row_chunk_size,
        )
    else:
        basis = precomputed_basis[: options.basis_count].contiguous()
        alpha = precomputed_alpha[:, : options.basis_count].contiguous()
    basis_device = basis.to(device)
    u = torch.empty(len(records), shape[0], options.rank, dtype=torch.float32)
    v = torch.empty(len(records), options.rank, shape[1], dtype=torch.float32)
    layer_errors: list[float] = []
    source_energy = 0.0
    difference_energy = 0.0
    residual_energy = 0.0
    for index, record in enumerate(records):
        source = store.load(record, device=device)["weight"].float()
        base = torch.einsum("k,koi->oi", alpha[index].to(device), basis_device)
        residual = source - base
        layer_u, layer_v = _layer_low_rank(residual, options.rank, options.seed + index)
        if options.rank:
            reconstruction = base + layer_u @ layer_v
            u[index].copy_(layer_u.cpu())
            v[index].copy_(layer_v.cpu())
        else:
            reconstruction = base
        source_norm = source.norm().clamp_min(1e-12)
        difference = reconstruction - source
        layer_errors.append((difference.norm() / source_norm).item())
        source_energy += source.square().sum().item()
        difference_energy += difference.square().sum().item()
        residual_energy += (reconstruction - base).square().sum().item()
        del source, base, residual, reconstruction, layer_u, layer_v
    original = len(records) * shape[0] * shape[1]
    compressed = basis.numel() + alpha.numel() + u.numel() + v.numel()
    alpha_probability = alpha.abs()
    alpha_probability /= alpha_probability.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(alpha_probability * alpha_probability.clamp_min(1e-12).log()).sum(1)
    entropy /= math.log(max(alpha.shape[1], 2))
    sorted_errors = torch.tensor(layer_errors)
    total_singular = eigenvalues.sum().clamp_min(1e-12)
    result = FactorizationResult(basis, alpha, u, v, {})
    result.metrics = {
        "options": asdict(options),
        "solver": "streamed_layer_gram_pca",
        "row_chunk_size": row_chunk_size,
        "original_params": original,
        "compressed_params": compressed,
        "compression_ratio": original / compressed,
        "original_bytes_float32": original * 4,
        "compressed_bytes_float32": compressed * 4,
        "weight_relative_error": math.sqrt(difference_energy / max(source_energy, 1e-12)),
        "layer_errors": layer_errors,
        "median_layer_error": sorted_errors.median().item(),
        "p95_layer_error": torch.quantile(sorted_errors, 0.95).item(),
        "max_layer_error": sorted_errors.max().item(),
        "singular_value_energy_coverage": (
            eigenvalues[: options.basis_count].sum() / total_singular
        ).item(),
        "basis_utilization": (alpha.square().sum(0).sqrt() > 1e-8).float().mean().item(),
        "alpha_entropy": entropy.mean().item(),
        "residual_energy_ratio": residual_energy / max(source_energy, 1e-12),
        "fit_seconds": perf_counter() - start_time,
    }
    return result
