from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .basis import BasisLinear, SharedBasisBank


def spatiotemporal_mask(
    frames: int, height: int, width: int, window: int, stride: int, device: torch.device
) -> torch.Tensor:
    """Local 3-D neighborhoods plus strided global anchor tokens."""
    n = frames * height * width
    ids = torch.arange(n, device=device)
    t = ids // (height * width)
    rem = ids % (height * width)
    y, x = rem // width, rem % width
    local = (
        ((t[:, None] - t[None]).abs() <= window)
        & ((y[:, None] - y[None]).abs() <= window)
        & ((x[:, None] - x[None]).abs() <= window)
    )
    anchors = (t % max(stride, 1) == 0) & (y % max(stride, 1) == 0) & (x % max(stride, 1) == 0)
    allowed = local | anchors[None, :] | torch.eye(n, device=device, dtype=torch.bool)
    return allowed


class SparseSelfAttention(nn.Module):
    def __init__(self, bank: SharedBasisBank, width: int, heads: int, rank: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.q, self.k, self.v, self.out = [BasisLinear(bank, width, rank) for _ in range(4)]

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> tuple[torch.Tensor, float]:
        b, n, _ = x.shape
        shape = (b, n, self.heads, self.head_dim)
        q = self.q(x).view(shape).transpose(1, 2)
        k = self.k(x).view(shape).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        bias = torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None, None])
        density = allowed.float().mean().item()
        return self.out(out.transpose(1, 2).reshape(b, n, -1)), density
