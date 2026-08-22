from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MotionBucket:
    name: str
    fraction: float
    keywords: tuple[str, ...]


def default_motion_buckets() -> list[MotionBucket]:
    return [
        MotionBucket("humans_animals", 0.20, ("person", "people", "man", "woman", "child", "dog", "cat", "animal", "bird")),
        MotionBucket("vehicles", 0.15, ("car", "truck", "train", "plane", "boat", "vehicle", "motorcycle", "bicycle")),
        MotionBucket("camera_motion", 0.15, ("camera", "pan", "zoom", "aerial", "drone", "tracking shot", "view moves")),
        MotionBucket("object_interaction", 0.15, ("holding", "cutting", "throwing", "cooking", "opening", "playing", "building", "pouring")),
        MotionBucket("fluid_elements", 0.10, ("water", "wave", "smoke", "fire", "flame", "steam", "rain", "river")),
        MotionBucket("nature", 0.10, ("forest", "mountain", "tree", "flower", "beach", "sky", "landscape", "nature")),
        MotionBucket("indoor", 0.10, ("room", "kitchen", "office", "house", "indoor", "table", "bedroom")),
        MotionBucket("high_dynamic", 0.05, ("running", "racing", "explosion", "fighting", "jumping", "dancing", "fast", "crowd")),
    ]


@dataclass(frozen=True)
class M3CorpusConfig:
    name: str = "mirage-m3-v0"
    source: str = "panda70m"
    source_metadata: str = "data/panda70m/train_2m.csv"
    work_dir: str = "data/mirage-m3-v0"
    downloaded_dir: str = "data/mirage-m3-v0/downloaded"
    normalized_dir: str = "data/mirage-m3-v0/normalized"
    shards_dir: str = "data/mirage-m3-v0/shards"
    target_clips: int = 10_000
    seed: int = 3407
    duration_min_s: float = 2.0
    duration_max_s: float = 8.0
    matching_score_min: float = 0.43
    max_clips_per_source: int = 3
    require_desirable: bool = True
    max_shot_segments: int = 1
    normalized_size: int = 256
    normalized_fps: int = 12
    audio_rate: int = 16_000
    min_download_width: int = 256
    min_download_height: int = 256
    require_audio: bool = True
    motion_difference_min: float = 0.015
    motion_difference_max: float = 0.65
    shard_size: int = 256
    validation_fraction: float = 0.02
    test_fraction: float = 0.02
    motion_buckets: list[MotionBucket] = field(default_factory=default_motion_buckets)

    @classmethod
    def from_json(cls, path: str | Path) -> M3CorpusConfig:
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        if "motion_buckets" in raw:
            raw["motion_buckets"] = [
                MotionBucket(
                    name=row["name"],
                    fraction=float(row["fraction"]),
                    keywords=tuple(row["keywords"]),
                )
                for row in raw["motion_buckets"]
            ]
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        if self.source != "panda70m":
            raise ValueError("M3-v0 currently accepts only the frozen Panda-70M source")
        if self.target_clips < 1 or self.shard_size < 1:
            raise ValueError("target_clips and shard_size must be positive")
        if not 0 < self.duration_min_s < self.duration_max_s:
            raise ValueError("invalid duration interval")
        if self.validation_fraction + self.test_fraction >= 0.5:
            raise ValueError("validation plus test fraction must stay below 0.5")
        if min(self.validation_fraction, self.test_fraction) < 0:
            raise ValueError("validation and test fractions cannot be negative")
        fraction = sum(bucket.fraction for bucket in self.motion_buckets)
        if abs(fraction - 1.0) > 1e-6:
            raise ValueError("motion bucket fractions must sum to one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
