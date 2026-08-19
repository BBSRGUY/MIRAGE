from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Self

import torch
from torch import nn


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    if hasattr(value, "sample") and isinstance(value.sample, torch.Tensor):
        return value.sample
    return None


class HookSet:
    """Owns PyTorch hook handles and guarantees explicit cleanup."""

    def __init__(self):
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture_module(
        self,
        module: nn.Module,
        name: str,
        callback: Callable[[str, torch.Tensor, str], None],
        capture_input: bool = True,
        capture_output: bool = True,
    ) -> None:
        if capture_input:

            def pre_hook(_module: nn.Module, args: tuple[Any, ...], _name: str = name) -> None:
                tensor = first_tensor(args)
                if tensor is not None:
                    callback(_name, tensor, "input")

            self._handles.append(module.register_forward_pre_hook(pre_hook))
        if capture_output:

            def post_hook(
                _module: nn.Module, _args: tuple[Any, ...], output: Any, _name: str = name
            ) -> None:
                tensor = first_tensor(output)
                if tensor is not None:
                    callback(_name, tensor, "output")

            self._handles.append(module.register_forward_hook(post_hook))

    def extend(self, handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
        self._handles.extend(handles)

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.clear()
