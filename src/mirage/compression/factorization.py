from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class FitOptions:
    basis_count: int
    rank: int
    initialization: str = "pca"
    optimization_steps: int = 0
    learning_rate: float = 1e-3
    alpha_sparsity: float = 0.0
    residual_magnitude: float = 0.0
    basis_orthogonality: float = 0.0
    seed: int = 0


@dataclass
class FactorizationResult:
    basis: torch.Tensor
    alpha: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    metrics: dict[str, Any]

    def reconstruct(self) -> torch.Tensor:
        base = torch.einsum("lk,koi->loi", self.alpha, self.basis)
        if self.u.shape[-1] == 0:
            return base
        return base + torch.einsum("lor,lri->loi", self.u, self.v)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "basis": self.basis.detach().cpu().contiguous(),
                "alpha": self.alpha.detach().cpu().contiguous(),
                "u": self.u.detach().cpu().contiguous(),
                "v": self.v.detach().cpu().contiguous(),
            },
            str(path),
        )
        path.with_suffix(".json").write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> FactorizationResult:
        path = Path(path)
        tensors = load_file(str(path), device=str(device))
        metrics = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(tensors["basis"], tensors["alpha"], tensors["u"], tensors["v"], metrics)


def compressed_parameter_count(
    layers: int, out_features: int, in_features: int, basis_count: int, rank: int
) -> int:
    return (
        basis_count * out_features * in_features
        + layers * basis_count
        + layers * rank * (out_features + in_features)
    )


def _orthonormalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(matrix.T, mode="reduced")
    return q.T


def _pca_basis(flat: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute right singular vectors through the small layer-by-layer Gram matrix."""
    gram = flat @ flat.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    valid = eigenvalues > torch.finfo(flat.dtype).eps * eigenvalues.max().clamp_min(1)
    usable = min(count, int(valid.sum().item()))
    basis = torch.zeros(count, flat.shape[1], device=flat.device, dtype=flat.dtype)
    if usable:
        basis[:usable] = (eigenvectors[:, :usable].T @ flat) / eigenvalues[:usable].sqrt()[
            :, None
        ].clamp_min(1e-12)
    return basis, eigenvalues


def _initial_basis(
    flat: torch.Tensor, count: int, strategy: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if strategy == "pca":
        return _pca_basis(flat, count)
    if strategy == "mean_residual":
        mean = flat.mean(dim=0, keepdim=True)
        mean = mean / mean.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if count == 1:
            return mean, torch.tensor([flat.square().sum()], device=flat.device)
        residual = flat - (flat @ mean.T) @ mean
        extra, values = _pca_basis(residual, count - 1)
        return torch.cat((mean, extra), dim=0), values
    if strategy == "random_orthogonal":
        generator = torch.Generator(device=flat.device).manual_seed(seed)
        mixing = torch.randn(
            count, flat.shape[0], generator=generator, device=flat.device, dtype=flat.dtype
        )
        basis = _orthonormalize_rows(mixing @ flat)
        return basis, torch.linalg.eigvalsh(flat @ flat.T).flip(0).clamp_min(0)
    raise ValueError(f"unknown initialization strategy: {strategy}")


def _coefficient_fit(flat: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    gram = basis @ basis.T
    return (flat @ basis.T) @ torch.linalg.pinv(gram)


def _low_rank(residual: torch.Tensor, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    layers, out_features, in_features = residual.shape
    if rank == 0:
        return (
            residual.new_empty((layers, out_features, 0)),
            residual.new_empty((layers, 0, in_features)),
        )
    u_values, v_values = [], []
    for index, matrix in enumerate(residual):
        effective = min(rank, min(matrix.shape))
        with torch.random.fork_rng(devices=[matrix.device] if matrix.is_cuda else []):
            torch.manual_seed(seed + index)
            if effective < min(matrix.shape) // 2:
                u, s, v = torch.svd_lowrank(matrix, q=effective, niter=4)
                vt = v.T
            else:
                u, s, vt = torch.linalg.svd(matrix, full_matrices=False)
                u, s, vt = u[:, :effective], s[:effective], vt[:effective]
        root = s.clamp_min(0).sqrt()
        u_values.append(u * root[None])
        v_values.append(root[:, None] * vt)
    return torch.stack(u_values), torch.stack(v_values)


def _metrics(
    weights: torch.Tensor,
    result: FactorizationResult,
    eigenvalues: torch.Tensor,
    elapsed: float,
    options: FitOptions,
) -> dict[str, Any]:
    reconstruction = result.reconstruct()
    differences = (reconstruction - weights).flatten(1).norm(dim=1)
    norms = weights.flatten(1).norm(dim=1).clamp_min(1e-12)
    layer_errors = differences / norms
    original = weights.numel()
    compressed = sum(t.numel() for t in (result.basis, result.alpha, result.u, result.v))
    alpha_probability = result.alpha.abs()
    alpha_probability = alpha_probability / alpha_probability.sum(dim=1, keepdim=True).clamp_min(
        1e-12
    )
    entropy = -(alpha_probability * alpha_probability.clamp_min(1e-12).log()).sum(1)
    entropy /= math.log(max(result.alpha.shape[1], 2))
    base = torch.einsum("lk,koi->loi", result.alpha, result.basis)
    residual_energy = (reconstruction - base).square().sum() / weights.square().sum().clamp_min(
        1e-12
    )
    total_singular = eigenvalues.sum().clamp_min(1e-12)
    coverage = eigenvalues[: options.basis_count].sum() / total_singular
    utilization = (result.alpha.square().sum(0).sqrt() > 1e-8).float().mean()
    return {
        "options": asdict(options),
        "original_params": original,
        "compressed_params": compressed,
        "compression_ratio": original / compressed,
        "original_bytes_float32": original * 4,
        "compressed_bytes_float32": compressed * 4,
        "weight_relative_error": torch.linalg.vector_norm(reconstruction - weights)
        .div(torch.linalg.vector_norm(weights).clamp_min(1e-12))
        .item(),
        "layer_errors": layer_errors.detach().cpu().tolist(),
        "median_layer_error": layer_errors.median().item(),
        "p95_layer_error": torch.quantile(layer_errors, 0.95).item(),
        "max_layer_error": layer_errors.max().item(),
        "singular_value_energy_coverage": coverage.item(),
        "basis_utilization": utilization.item(),
        "alpha_entropy": entropy.mean().item(),
        "residual_energy_ratio": residual_energy.item(),
        "fit_seconds": elapsed,
    }


def fit_shared_basis(weights: torch.Tensor, options: FitOptions) -> FactorizationResult:
    """Fit shared bases and deterministic per-layer low-rank residuals."""
    if weights.ndim != 3:
        raise ValueError("weights must have shape [layers, out_features, in_features]")
    layers, out_features, in_features = weights.shape
    if options.basis_count < 1:
        raise ValueError("basis_count must be positive")
    if options.basis_count > layers:
        raise ValueError(f"basis_count {options.basis_count} exceeds layer count {layers}")
    if options.rank < 0 or options.rank > min(out_features, in_features):
        raise ValueError(
            f"rank {options.rank} invalid for matrix shape {(out_features, in_features)}"
        )
    start = perf_counter()
    source = weights.detach().float()
    flat = source.flatten(1)
    basis_flat, eigenvalues = _initial_basis(
        flat, options.basis_count, options.initialization, options.seed
    )
    alpha = _coefficient_fit(flat, basis_flat)
    basis = basis_flat.view(options.basis_count, out_features, in_features)
    residual = source - torch.einsum("lk,koi->loi", alpha, basis)
    u, v = _low_rank(residual, options.rank, options.seed)

    if options.optimization_steps:
        basis = torch.nn.Parameter(basis)
        alpha = torch.nn.Parameter(alpha)
        u = torch.nn.Parameter(u)
        v = torch.nn.Parameter(v)
        optimizer = torch.optim.Adam([basis, alpha, u, v], lr=options.learning_rate)
        denominator = source.square().mean().clamp_min(1e-12)
        for _ in range(options.optimization_steps):
            optimizer.zero_grad(set_to_none=True)
            reconstructed = torch.einsum("lk,koi->loi", alpha, basis)
            if options.rank:
                reconstructed = reconstructed + torch.einsum("lor,lri->loi", u, v)
            loss = (reconstructed - source).square().mean() / denominator
            loss = loss + options.alpha_sparsity * alpha.abs().mean()
            if options.rank:
                loss = loss + options.residual_magnitude * (u.square().mean() + v.square().mean())
            if options.basis_orthogonality:
                normalized = torch.nn.functional.normalize(basis.flatten(1), dim=1)
                identity = torch.eye(options.basis_count, device=source.device)
                loss = (
                    loss
                    + options.basis_orthogonality
                    * (normalized @ normalized.T - identity).square().mean()
                )
            loss.backward()
            optimizer.step()
        basis, alpha, u, v = (item.detach() for item in (basis, alpha, u, v))

    result = FactorizationResult(basis, alpha, u, v, {})
    result.metrics = _metrics(source, result, eigenvalues, perf_counter() - start, options)
    return result
