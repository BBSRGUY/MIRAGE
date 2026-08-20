from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch
from torch import nn

CaptureCallback = Callable[[str, torch.Tensor, dict[str, Any]], None]


class TeacherAdapter(ABC):
    """Teacher-neutral interface used by all offline M2 experiments."""

    @property
    @abstractmethod
    def model_identifier(self) -> str: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def block_count(self) -> int: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def named_projections(self) -> Mapping[str, nn.Linear]: ...

    @abstractmethod
    def named_projection_names(self) -> tuple[str, ...]: ...

    @abstractmethod
    def iter_projection_tensors(
        self, names: set[str] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]: ...

    @abstractmethod
    def install_capture_hooks(self, callback: CaptureCallback) -> None: ...

    @abstractmethod
    def run_prompt(
        self,
        prompt: str,
        *,
        sample_id: str,
        split: str,
        seed: int,
        frames: int,
        height: int,
        width: int,
        steps: int,
        guidance_scale: float,
        max_sequence_length: int,
    ) -> None: ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...

    @abstractmethod
    def unload(self) -> None: ...
