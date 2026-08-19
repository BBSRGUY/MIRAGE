from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import torch


@dataclass
class RunTelemetry:
    latency_s: float = 0.0
    peak_vram_bytes: int = 0
    peak_reserved_vram_bytes: int = 0
    allocated_vram_bytes: int = 0
    parameter_bytes: int = 0
    estimated_flops: int = 0
    attention_density: float = 1.0
    cache_hits: int = 0
    cache_queries: int = 0
    precision_counts: dict[str, int] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.cache_queries if self.cache_queries else 0.0

    def record_precision(self, name: str) -> None:
        self.precision_counts[name] = self.precision_counts.get(name, 0) + 1

    def to_dict(self) -> dict:
        value = asdict(self)
        value["cache_hit_rate"] = self.cache_hit_rate
        return value

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class Measure:
    def __init__(self, telemetry: RunTelemetry, device: torch.device):
        self.telemetry = telemetry
        self.device = device

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        self.start = perf_counter()
        return self

    def __exit__(self, *_):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.telemetry.peak_vram_bytes = torch.cuda.max_memory_allocated(self.device)
            self.telemetry.peak_reserved_vram_bytes = torch.cuda.max_memory_reserved(self.device)
            self.telemetry.allocated_vram_bytes = torch.cuda.memory_allocated(self.device)
        self.telemetry.latency_s = perf_counter() - self.start


def parameter_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def estimate_resident_bytes(module: torch.nn.Module, config, batch_size: int = 1) -> int:
    """Conservative parameter + live activation estimate, not allocator reservation."""
    params = parameter_bytes(module)
    bytes_per = 4 if config.precision == "fp32" else 2
    tokens = config.video_tokens + config.text_tokens
    activations = batch_size * tokens * config.hidden_size * bytes_per
    qkv = 3 * activations
    codec = batch_size * config.frames * config.height * config.width * 3 * 4
    cache = config.depth * batch_size * config.video_tokens * config.hidden_size * bytes_per
    return params + activations + qkv + codec + cache
