from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MirageConfig:
    """Shape and runtime contract for one MIRAGE experiment."""

    frames: int = 8
    height: int = 64
    width: int = 64
    patch_size: int = 8
    hidden_size: int = 192
    depth: int = 8
    heads: int = 6
    basis_count: int = 4
    residual_rank: int = 8
    projection_backend: str = "shared_basis"
    steps: int = 4
    text_tokens: int = 32
    vocabulary_size: int = 8192
    scene_ratio: float = 0.5
    attention_window: int = 1
    attention_stride: int = 4
    cache_threshold: float = 0.08
    max_cache_age: int = 2
    precision: str = "dynamic"
    seed: int = 0
    vram_budget_gb: float = 24.0

    @property
    def latent_height(self) -> int:
        return self.height // self.patch_size

    @property
    def latent_width(self) -> int:
        return self.width // self.patch_size

    @property
    def video_tokens(self) -> int:
        return self.frames * self.latent_height * self.latent_width

    def validate(self) -> None:
        if self.height % self.patch_size or self.width % self.patch_size:
            raise ValueError("height and width must be divisible by patch_size")
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if not 0.0 < self.scene_ratio < 1.0:
            raise ValueError("scene_ratio must be between 0 and 1")
        if self.steps < 1 or self.depth < 1 or self.frames < 1:
            raise ValueError("steps, depth, and frames must be positive")
        if self.precision not in {"dynamic", "fp32", "bf16"}:
            raise ValueError("precision must be dynamic, fp32, or bf16")
        if self.projection_backend not in {"independent", "shared_basis"}:
            raise ValueError("projection_backend must be independent or shared_basis")

    @classmethod
    def from_json(cls, path: str | Path) -> MirageConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)
