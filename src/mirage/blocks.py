from __future__ import annotations

import torch
from torch import nn

from .attention import IndependentSparseSelfAttention, SparseSelfAttention
from .basis import BasisLinear, SharedBasisBank
from .telemetry import RunTelemetry


class MirageBlock(nn.Module):
    def __init__(
        self,
        bank: SharedBasisBank,
        width: int,
        heads: int,
        rank: int,
        cache_threshold: float,
        max_cache_age: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.attn = SparseSelfAttention(bank, width, heads, rank)
        self.ff1 = BasisLinear(bank, width, rank)
        self.ff2 = BasisLinear(bank, width, rank)
        self.activation = nn.SiLU()
        self.modulation = nn.Linear(width, width * 2)
        self.cache_threshold = cache_threshold
        self.max_cache_age = max_cache_age
        self._cached_signature: torch.Tensor | None = None
        self._cached_residual: torch.Tensor | None = None
        self._cache_age = 0

    def clear_cache(self) -> None:
        self._cached_signature = None
        self._cached_residual = None
        self._cache_age = 0

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        allowed: torch.Tensor,
        telemetry: RunTelemetry,
        allow_cache: bool,
    ) -> torch.Tensor:
        telemetry.cache_queries += int(allow_cache)
        signature = x.detach().float().mean(dim=1, keepdim=True)
        can_reuse = False
        if allow_cache and self._cached_signature is not None and self._cached_residual is not None:
            change = (signature - self._cached_signature).square().mean().sqrt()
            scale = self._cached_signature.square().mean().sqrt().clamp_min(1e-5)
            can_reuse = bool(
                change / scale < self.cache_threshold and self._cache_age < self.max_cache_age
            )
        if can_reuse:
            telemetry.cache_hits += 1
            self._cache_age += 1
            return x + self._cached_residual

        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale[:, None]) + shift[:, None]
        attention, density = self.attn(h, allowed)
        h = x + attention
        residual = self.ff2(self.activation(self.ff1(self.norm2(h))))
        output = h + residual
        self._cached_signature = signature
        self._cached_residual = (output - x).detach()
        self._cache_age = 0
        telemetry.attention_density = min(telemetry.attention_density, density)
        return output


class DenseBlock(nn.Module):
    """Independent-weight reference block for controlled ablations."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width))
        self.modulation = nn.Linear(width, width * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale[:, None]) + shift[:, None]
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class IndependentMirageBlock(MirageBlock):
    """MIRAGE block with independent projections and the same sparse/cache behavior."""

    def __init__(
        self, width: int, heads: int, cache_threshold: float, max_cache_age: int
    ) -> None:
        nn.Module.__init__(self)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.attn = IndependentSparseSelfAttention(width, heads)
        self.ff1 = nn.Linear(width, width * 4)
        self.ff2 = nn.Linear(width * 4, width)
        self.activation = nn.SiLU()
        self.modulation = nn.Linear(width, width * 2)
        self.cache_threshold = cache_threshold
        self.max_cache_age = max_cache_age
        self._cached_signature = None
        self._cached_residual = None
        self._cache_age = 0
