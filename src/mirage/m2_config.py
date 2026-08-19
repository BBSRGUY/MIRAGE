from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherConfig:
    type: str = "ltx25"
    model_id: str = "Lightricks/LTX-2.5"
    revision: str | None = None
    model_root: str | None = None
    transformer_file: str | None = None
    text_encoder_file: str | None = None
    video_vae_file: str | None = None
    audio_vae_file: str | None = None
    spatial_upsampler_file: str | None = None
    duration_head_file: str | None = None
    ltx_repo_path: str = ".vendor/LTX-2"
    offload_mode: str = "disk"
    dtype: str = "bfloat16"
    device: str = "cuda"
    local_files_only: bool = False
    max_capture_tokens: int = 256
    capture_blocks: list[int] | None = None


@dataclass(frozen=True)
class DataConfig:
    prompts_file: str = "configs/m2_prompts.txt"
    num_train_prompts: int = 4
    num_eval_prompts: int = 2
    seed: int = 1337


@dataclass(frozen=True)
class GenerationConfig:
    frames: int = 9
    height: int = 256
    width: int = 256
    steps: int = 8
    guidance_scale: float = 1.0
    max_sequence_length: int = 128


@dataclass(frozen=True)
class CompressionConfig:
    basis_counts: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    ranks: list[int] = field(default_factory=lambda: [0, 4, 8, 16, 32, 64])
    initialization: str = "pca"
    fit_steps: int = 0
    activation_fit_steps: int = 100
    learning_rate: float = 1e-3
    lambda_weight: float = 0.1
    lambda_activation: float = 1.0
    alpha_sparsity: float = 0.0
    residual_magnitude: float = 0.0
    basis_orthogonality: float = 0.0


@dataclass(frozen=True)
class TemporalConfig:
    reuse_thresholds: list[float] = field(
        default_factory=lambda: [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]
    )
    reuse_threshold: float = 0.02
    predict_threshold: float = 0.05
    predictor_steps: int = 200
    predictor_learning_rate: float = 1e-3


@dataclass(frozen=True)
class SceneMotionConfig:
    pca_ranks: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    lowpass_alpha: float = 0.8


@dataclass(frozen=True)
class AcceptanceConfig:
    validation_activation_cosine: float = 0.995
    normalized_activation_error: float = 0.05
    compression_ratio: float = 3.0
    reuse_predict_coverage: float = 0.4
    scene_low_rank_energy: float = 0.6


@dataclass(frozen=True)
class M2Config:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    scene_motion: SceneMotionConfig = field(default_factory=SceneMotionConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    output_dir: str = "artifacts/m2/ltx_study"

    @classmethod
    def from_json(cls, path: str | Path) -> M2Config:
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            teacher=TeacherConfig(**raw.get("teacher", {})),
            data=DataConfig(**raw.get("data", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
            compression=CompressionConfig(**raw.get("compression", {})),
            temporal=TemporalConfig(**raw.get("temporal", {})),
            scene_motion=SceneMotionConfig(**raw.get("scene_motion", {})),
            acceptance=AcceptanceConfig(**raw.get("acceptance", {})),
            output_dir=raw.get("output_dir", "artifacts/m2/ltx_study"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.teacher.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported teacher dtype: {self.teacher.dtype}")
        if self.teacher.type != "ltx25":
            raise ValueError(f"unsupported teacher type: {self.teacher.type}")
        if self.teacher.offload_mode not in {"cpu", "disk"}:
            raise ValueError("LTX-2.5 22B requires cpu or disk block streaming for a 24 GB GPU")
        if self.generation.frames < 1 or self.generation.steps < 1:
            raise ValueError("frames and steps must be positive")
        if self.data.num_train_prompts < 1 or self.data.num_eval_prompts < 1:
            raise ValueError("train and evaluation prompt counts must be positive")
        if self.compression.initialization not in {"pca", "random_orthogonal", "mean_residual"}:
            raise ValueError("unknown compression initialization")
