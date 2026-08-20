from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import torch
from safetensors.torch import load_file, save_file

from ..datasets import FeatureRecord, FeatureStore
from .streamed_fit import _layer_low_rank, _weight_rows, streamed_pca_basis


@dataclass
class HierarchicalFactorizationResult:
    global_basis: torch.Tensor
    global_alpha: torch.Tensor
    group_basis: torch.Tensor
    group_beta: torch.Tensor
    group_ids: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    metrics: dict[str, Any]

    @property
    def layer_count(self) -> int:
        return self.global_alpha.shape[0]

    def reconstruct_layer(self, index: int) -> torch.Tensor:
        value = self.reconstruct_base_layer(index)
        rank = int(self.metrics["rank_per_layer"][index])
        if rank:
            value = value + self.u[index, :, :rank] @ self.v[index, :rank]
        return value

    def reconstruct_base_layer(self, index: int) -> torch.Tensor:
        group = int(self.group_ids[index].item())
        value = torch.einsum("k,koi->oi", self.global_alpha[index], self.global_basis)
        value = value + torch.einsum(
            "k,koi->oi", self.group_beta[index], self.group_basis[group]
        )
        return value

    def fit_activation_residual(
        self,
        index: int,
        source: torch.Tensor,
        inputs: torch.Tensor,
        *,
        ridge: float,
        seed: int,
    ) -> dict[str, float]:
        """Fit the fixed-rank layer residual in the observed training-activation metric."""
        rank = int(self.metrics["rank_per_layer"][index])
        if rank == 0:
            return {"rank": 0, "training_output_error": 1.0}
        base = self.reconstruct_base_layer(index)
        x = inputs.float()
        target = x @ (source.float() - base).T
        effective = min(rank, min(target.shape))
        with torch.random.fork_rng(devices=[target.device] if target.is_cuda else []):
            torch.manual_seed(seed)
            left, singular, right = torch.svd_lowrank(target, q=effective, niter=2)
        root = singular.clamp_min(0).sqrt()
        activation_factor = left * root[None]
        output_factor = right * root[None]
        gram = x @ x.T
        scale = gram.diagonal().mean().clamp_min(1e-12)
        gram.diagonal().add_(ridge * scale)
        coefficients = torch.linalg.solve(gram, activation_factor)
        input_factor = (x.T @ coefficients).T
        self.u[index].zero_()
        self.v[index].zero_()
        self.u[index, :, :effective].copy_(output_factor)
        self.v[index, :effective].copy_(input_factor)
        predicted = x @ (output_factor @ input_factor).T
        error = (predicted - target).norm() / target.norm().clamp_min(1e-12)
        return {"rank": effective, "training_output_error": error.item()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "global_basis": self.global_basis.detach().cpu().contiguous(),
                "global_alpha": self.global_alpha.detach().cpu().contiguous(),
                "group_basis": self.group_basis.detach().cpu().contiguous(),
                "group_beta": self.group_beta.detach().cpu().contiguous(),
                "group_ids": self.group_ids.detach().cpu().contiguous(),
                "u": self.u.detach().cpu().contiguous(),
                "v": self.v.detach().cpu().contiguous(),
            },
            str(path),
        )
        path.with_suffix(".json").write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> HierarchicalFactorizationResult:
        path = Path(path)
        tensors = load_file(str(path), device=str(device))
        metrics = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(metrics=metrics, **tensors)


def _group_pca(
    store: FeatureStore,
    records: Sequence[FeatureRecord],
    indices: list[int],
    shape: tuple[int, int],
    global_basis: torch.Tensor,
    global_alpha: torch.Tensor,
    count: int,
    device: torch.device,
    row_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gram = torch.zeros(len(indices), len(indices), device=device, dtype=torch.float64)
    global_device = global_basis.to(device)
    for start in range(0, shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, shape[0])
        weights = torch.stack(
            [_weight_rows(store, records[index], start, stop, device) for index in indices]
        )
        global_rows = global_device[:, start:stop]
        residual = weights - torch.einsum(
            "lk,koi->loi", global_alpha[indices].to(device), global_rows
        )
        flat = residual.flatten(1)
        gram.addmm_(flat.double(), flat.double().T)
        del weights, residual, flat
    values, vectors = torch.linalg.eigh(gram.float())
    order = torch.argsort(values, descending=True)
    values, vectors = values[order].clamp_min(0), vectors[:, order]
    usable = min(count, int((values > 1e-8 * values.max()).sum().item()))
    basis = torch.zeros(count, shape[0], shape[1], dtype=torch.float32)
    beta = torch.zeros(len(indices), count, dtype=torch.float32)
    if usable:
        mixing = vectors[:, :usable].T / values[:usable].sqrt()[:, None].clamp_min(1e-12)
        beta[:, :usable] = (vectors[:, :usable] * values[:usable].sqrt()[None]).cpu()
        for start in range(0, shape[0], row_chunk_size):
            stop = min(start + row_chunk_size, shape[0])
            weights = torch.stack(
                [_weight_rows(store, records[index], start, stop, device) for index in indices]
            )
            residual = weights - torch.einsum(
                "lk,koi->loi",
                global_alpha[indices].to(device),
                global_device[:, start:stop],
            )
            basis[:usable, start:stop].copy_(
                torch.einsum("kl,loi->koi", mixing, residual).cpu()
            )
            del weights, residual
    return basis, beta


def fit_hierarchical_shared_basis(
    store: FeatureStore,
    records: Sequence[FeatureRecord],
    shape: tuple[int, int],
    *,
    groups: Sequence[Sequence[int]],
    global_basis_count: int,
    group_basis_count: int,
    rank_per_layer: Sequence[int],
    device: torch.device,
    row_chunk_size: int,
    seed: int,
) -> HierarchicalFactorizationResult:
    if sorted(index for group in groups for index in group) != list(range(len(records))):
        raise ValueError("groups must partition all layers exactly once")
    if len(rank_per_layer) != len(records):
        raise ValueError("rank_per_layer must match the layer count")
    start_time = perf_counter()
    global_basis, global_alpha, eigenvalues = streamed_pca_basis(
        store,
        records,
        shape,
        basis_count=global_basis_count,
        device=device,
        row_chunk_size=row_chunk_size,
    )
    group_basis = torch.empty(
        len(groups), group_basis_count, shape[0], shape[1], dtype=torch.float32
    )
    group_beta = torch.zeros(len(records), group_basis_count, dtype=torch.float32)
    group_ids = torch.empty(len(records), dtype=torch.int64)
    for group_id, group_values in enumerate(groups):
        indices = list(group_values)
        basis, beta = _group_pca(
            store,
            records,
            indices,
            shape,
            global_basis,
            global_alpha,
            group_basis_count,
            device,
            row_chunk_size,
        )
        group_basis[group_id].copy_(basis)
        group_beta[indices] = beta
        group_ids[indices] = group_id
    maximum_rank = max(rank_per_layer, default=0)
    u = torch.zeros(len(records), shape[0], maximum_rank, dtype=torch.float32)
    v = torch.zeros(len(records), maximum_rank, shape[1], dtype=torch.float32)
    global_device = global_basis.to(device)
    groups_device = group_basis.to(device)
    layer_errors: list[float] = []
    source_energy = difference_energy = residual_energy = 0.0
    for index, record in enumerate(records):
        source = store.load(record, device=device)["weight"].float()
        group_id = int(group_ids[index].item())
        base = torch.einsum("k,koi->oi", global_alpha[index].to(device), global_device)
        base = base + torch.einsum(
            "k,koi->oi", group_beta[index].to(device), groups_device[group_id]
        )
        rank = int(rank_per_layer[index])
        layer_u, layer_v = _layer_low_rank(source - base, rank, seed + index)
        reconstruction = base
        if rank:
            reconstruction = reconstruction + layer_u @ layer_v
            u[index, :, :rank].copy_(layer_u.cpu())
            v[index, :rank].copy_(layer_v.cpu())
        difference = reconstruction - source
        source_norm = source.norm().clamp_min(1e-12)
        layer_errors.append((difference.norm() / source_norm).item())
        source_energy += source.square().sum().item()
        difference_energy += difference.square().sum().item()
        residual_energy += (reconstruction - base).square().sum().item()
        del source, base, reconstruction, difference, layer_u, layer_v
    original = len(records) * math.prod(shape)
    compressed = (
        global_basis.numel()
        + global_alpha.numel()
        + group_basis.numel()
        + group_beta.numel()
        + sum(rank * (shape[0] + shape[1]) for rank in rank_per_layer)
    )
    errors = torch.tensor(layer_errors)
    metrics: dict[str, Any] = {
        "format": "hierarchical_shared_basis_v1",
        "solver": "streamed_hierarchical_layer_gram_pca",
        "global_basis_count": global_basis_count,
        "group_basis_count": group_basis_count,
        "groups": [list(group) for group in groups],
        "rank_per_layer": list(rank_per_layer),
        "row_chunk_size": row_chunk_size,
        "original_params": original,
        "compressed_params": compressed,
        "compression_ratio": original / compressed,
        "weight_relative_error": math.sqrt(difference_energy / max(source_energy, 1e-12)),
        "layer_errors": layer_errors,
        "median_layer_error": errors.median().item(),
        "p95_layer_error": torch.quantile(errors, 0.95).item(),
        "max_layer_error": errors.max().item(),
        "global_singular_energy_coverage": (
            eigenvalues[:global_basis_count].sum() / eigenvalues.sum().clamp_min(1e-12)
        ).item(),
        "residual_energy_ratio": residual_energy / max(source_energy, 1e-12),
        "fit_seconds": perf_counter() - start_time,
    }
    return HierarchicalFactorizationResult(
        global_basis,
        global_alpha,
        group_basis,
        group_beta,
        group_ids,
        u,
        v,
        metrics,
    )
