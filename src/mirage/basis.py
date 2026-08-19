from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SharedBasisBank(nn.Module):
    """A bank of transformations reused by every layer in a stack."""

    def __init__(self, basis_count: int, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(basis_count, width, width))
        nn.init.normal_(self.weight, std=0.02 / math.sqrt(basis_count))


class BasisLinear(nn.Module):
    """W_l = sum(alpha_l,i B_i) + U_l V_l, composed only for the current op."""

    def __init__(self, bank: SharedBasisBank, width: int, rank: int):
        super().__init__()
        self.bank = bank
        self.alpha = nn.Parameter(torch.zeros(bank.weight.shape[0]))
        self.u = nn.Parameter(torch.empty(width, rank))
        self.v = nn.Parameter(torch.empty(rank, width))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.alpha, std=1 / math.sqrt(bank.weight.shape[0]))
        nn.init.normal_(self.u, std=0.01)
        nn.init.normal_(self.v, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.einsum("k,koi->oi", self.alpha, self.bank.weight)
        return F.linear(x, base, self.bias) + (x @ self.v.t()) @ self.u.t()
