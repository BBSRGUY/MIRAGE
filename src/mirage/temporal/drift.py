from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DriftMetrics:
    normalized_drift: float
    cosine_similarity: float
    centered_correlation: float
    residual_rms: float
    mean_token_drift: float
    p95_token_drift: float
    max_token_drift: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def drift_metrics(
    previous: torch.Tensor, current: torch.Tensor, epsilon: float = 1e-8
) -> DriftMetrics:
    """Compare adjacent-step features with scale-normalized and tokenwise metrics."""
    if previous.shape != current.shape:
        raise ValueError(f"temporal feature shapes differ: {previous.shape} vs {current.shape}")
    a, b = previous.float(), current.float()
    difference = b - a
    rms_a = a.square().mean().sqrt()
    rms_b = b.square().mean().sqrt()
    normalized = difference.square().mean().sqrt() / torch.maximum(rms_a, rms_b).clamp_min(epsilon)
    cosine = F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    centered_a, centered_b = a - a.mean(), b - b.mean()
    correlation = F.cosine_similarity(centered_a.flatten(), centered_b.flatten(), dim=0)
    if a.ndim >= 2:
        token_difference = difference.reshape(-1, difference.shape[-1]).square().mean(-1).sqrt()
        token_scale = torch.maximum(
            a.reshape(-1, a.shape[-1]).square().mean(-1).sqrt(),
            b.reshape(-1, b.shape[-1]).square().mean(-1).sqrt(),
        ).clamp_min(epsilon)
        token_drift = token_difference / token_scale
    else:
        token_drift = difference.abs() / torch.maximum(a.abs(), b.abs()).clamp_min(epsilon)
    return DriftMetrics(
        normalized_drift=normalized.item(),
        cosine_similarity=cosine.item(),
        centered_correlation=correlation.item(),
        residual_rms=difference.square().mean().sqrt().item(),
        mean_token_drift=token_drift.mean().item(),
        p95_token_drift=torch.quantile(token_drift, 0.95).item(),
        max_token_drift=token_drift.max().item(),
    )
