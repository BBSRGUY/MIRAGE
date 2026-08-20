from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import MirageConfig


@dataclass(frozen=True)
class M3DataConfig:
    manifest: str = "synthetic://mirage-smoke"
    teacher_features: str | None = None
    shuffle_buffer: int = 32
    synthetic_samples: int = 64
    seed: int = 3407


@dataclass(frozen=True)
class M3CompressionConfig:
    policy: str = "m2_heterogeneous_int4_int8"
    allocation_report: str = "artifacts/m2/ltx25_22b/reports/heterogeneous_allocation.json"
    int4_group_size: int = 64


@dataclass(frozen=True)
class M3TrainingConfig:
    output_dir: str = "artifacts/m3/smoke"
    batch_size: int = 2
    epochs: int = 1
    max_steps: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_accumulation: int = 1
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    ema_decay: float = 0.999
    checkpoint_every: int = 10
    log_every: int = 1
    max_grad_norm: float = 1.0
    resume: str | None = None
    strict_resume_provenance: bool = True
    seed: int = 3407


@dataclass(frozen=True)
class M3LossConfig:
    flow: float = 1.0
    behavior: float = 0.1
    temporal: float = 0.1
    identity: float = 0.05
    av_sync: float = 0.05


@dataclass(frozen=True)
class M3EvalConfig:
    prompts: list[str] = field(
        default_factory=lambda: [
            "A glass sphere rolls across a wooden table.",
            "A ceramic robot makes tea in a warm kitchen.",
        ]
    )
    seeds: list[int] = field(default_factory=lambda: [101, 202])


@dataclass(frozen=True)
class M3Config:
    variant: str = "MIRAGE-S"
    model: MirageConfig = field(
        default_factory=lambda: MirageConfig(
            projection_backend="independent", cache_threshold=0.0, vram_budget_gb=20.0
        )
    )
    data: M3DataConfig = field(default_factory=M3DataConfig)
    compression: M3CompressionConfig = field(default_factory=M3CompressionConfig)
    training: M3TrainingConfig = field(default_factory=M3TrainingConfig)
    losses: M3LossConfig = field(default_factory=M3LossConfig)
    evaluation: M3EvalConfig = field(default_factory=M3EvalConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> M3Config:
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            variant=raw.get("variant", "MIRAGE-S"),
            model=MirageConfig(**raw.get("model", {})),
            data=M3DataConfig(**raw.get("data", {})),
            compression=M3CompressionConfig(**raw.get("compression", {})),
            training=M3TrainingConfig(**raw.get("training", {})),
            losses=M3LossConfig(**raw.get("losses", {})),
            evaluation=M3EvalConfig(**raw.get("evaluation", {})),
        )

    def validate(self) -> None:
        self.model.validate()
        if self.variant not in {"MIRAGE-S", "MIRAGE-M", "MIRAGE-L"}:
            raise ValueError("variant must be MIRAGE-S, MIRAGE-M, or MIRAGE-L")
        if self.training.mixed_precision not in {"fp32", "bf16"}:
            raise ValueError("M3 mixed_precision must be fp32 or bf16")
        if min(self.training.batch_size, self.training.max_steps) < 1:
            raise ValueError("batch_size and max_steps must be positive")
        if self.compression.policy == "m2_heterogeneous_int4_int8":
            if self.model.projection_backend != "independent":
                raise ValueError("the default M2 INT4/INT8 policy requires independent projections")
            if self.model.cache_threshold != 0:
                raise ValueError("default M3 configs must disable cache/predict execution")
        elif self.compression.policy != "shared_basis_ablation":
            raise ValueError("unsupported M3 compression policy")
        if self.model.vram_budget_gb >= 24:
            raise ValueError("M3 residency budget must leave headroom below physical 24 GiB")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
