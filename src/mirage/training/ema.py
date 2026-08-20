from __future__ import annotations

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: value.detach().float().cpu().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach().float().cpu(), 1 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.shadow = state["shadow"]  # type: ignore[assignment]

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))
