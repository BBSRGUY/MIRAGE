from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class QuantizedLinear(nn.Module):
    """Reference resident INT4/INT8 linear; kernels can replace dequantization in M4."""

    def __init__(self, linear: nn.Linear, precision: str, group_size: int = 64):
        super().__init__()
        weight = linear.weight.detach().float()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.precision = precision
        self.group_size = group_size
        if precision == "INT8_ROW":
            scale = weight.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
            self.register_buffer("quantized", (weight / scale).round().clamp(-127, 127).to(torch.int8))
            self.register_buffer("scale", scale.to(torch.bfloat16))
        elif precision == "INT4_GROUP64":
            if self.in_features % group_size:
                raise ValueError(f"input width {self.in_features} is not divisible by {group_size}")
            grouped = weight.reshape(self.out_features, self.in_features // group_size, group_size)
            scale = grouped.abs().amax(2, keepdim=True).clamp_min(1e-12) / 7
            values = (grouped / scale).round().clamp(-7, 7).to(torch.int16) + 8
            flat = values.to(torch.uint8).flatten()
            packed = flat[0::2] | (flat[1::2] << 4)
            self.register_buffer("quantized", packed)
            self.register_buffer("scale", scale.to(torch.bfloat16))
        else:
            raise ValueError(f"unsupported projection precision: {precision}")
        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", linear.bias.detach().to(torch.bfloat16))

    def _weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.precision == "INT8_ROW":
            return (self.quantized.float() * self.scale.float()).to(dtype)
        low = self.quantized & 0x0F
        high = self.quantized >> 4
        values = torch.stack((low, high), dim=1).flatten().to(torch.int16) - 8
        grouped = values.reshape(self.out_features, self.in_features // self.group_size, self.group_size)
        return (grouped.float() * self.scale.float()).reshape(
            self.out_features, self.in_features
        ).to(dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(value.dtype) if self.bias is not None else None
        return F.linear(value, self._weight(value.dtype), bias)


def _policy_map(depth: int, allocation_report: str | Path) -> dict[tuple[int, str], str]:
    report = json.loads(Path(allocation_report).read_text(encoding="utf-8"))
    decisions = {
        (int(row["layer"].split(".")[1]), row["family"]): row["choice"]
        for row in report["portfolio"]["decisions"]
    }
    result = {}
    for index in range(depth):
        teacher_index = round(index * 47 / max(depth - 1, 1))
        for family in ("attn.q", "attn.k", "attn.v", "attn.out", "ff-in", "ff-out"):
            result[(index, family)] = decisions[(teacher_index, family)]
    return result


def apply_m2_mixed_precision(
    model: nn.Module, allocation_report: str | Path, group_size: int = 64
) -> dict[str, Any]:
    if getattr(model.config, "projection_backend", None) != "independent":
        raise ValueError("M2 mixed precision applies only to the independent M3 baseline")
    policy = _policy_map(len(model.blocks), allocation_report)
    counts: dict[str, int] = {}
    for index, block in enumerate(model.blocks):
        targets = {
            "attn.q": (block.attn, "q"),
            "attn.k": (block.attn, "k"),
            "attn.v": (block.attn, "v"),
            "attn.out": (block.attn, "out"),
            "ff-in": (block, "ff1"),
            "ff-out": (block, "ff2"),
        }
        for family, (parent, attribute) in targets.items():
            precision = policy[(index, family)]
            linear = getattr(parent, attribute)
            setattr(parent, attribute, QuantizedLinear(linear, precision, group_size))
            counts[precision] = counts.get(precision, 0) + 1
    stored_bytes = sum(
        value.numel() * value.element_size() for value in model.state_dict().values()
    )
    return {
        "policy": "m2_heterogeneous_int4_int8",
        "representation_counts": counts,
        "resident_state_bytes": stored_bytes,
        "teacher_dependency": False,
        "cpu_offload": False,
    }
