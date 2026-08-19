from __future__ import annotations

from contextlib import nullcontext

import torch


class DynamicPrecisionPolicy:
    """Executable BF16/FP32 schedule; low-bit kernels are a later milestone."""

    def __init__(self, mode: str, steps: int):
        self.mode = mode
        self.steps = steps

    def dtype_for(self, step: int, device: torch.device) -> torch.dtype:
        if self.mode == "fp32" or device.type != "cuda":
            return torch.float32
        if self.mode == "bf16":
            return torch.bfloat16
        edge = step == 0 or step == self.steps - 1
        return torch.float32 if edge else torch.bfloat16

    def context(self, step: int, device: torch.device):
        dtype = self.dtype_for(step, device)
        if device.type == "cuda" and dtype != torch.float32:
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()
