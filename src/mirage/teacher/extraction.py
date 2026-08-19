from __future__ import annotations

import json
import platform
import subprocess
from typing import Any

import torch

from ..datasets import FeatureStore, PromptRecord
from ..m2_config import M2Config
from .base import TeacherAdapter


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class TeacherExtractor:
    """Streams real teacher weights and hooked features into a resumable store."""

    def __init__(self, adapter: TeacherAdapter, store: FeatureStore, config: M2Config):
        self.adapter = adapter
        self.store = store
        self.config = config
        self._record_counts: dict[str, int] = {}

    def _capture(self, name: str, tensor: torch.Tensor, metadata: dict[str, Any]) -> None:
        sample_id = str(metadata["sample_id"])
        split = str(metadata["split"])
        step = int(metadata.get("step_index", -1))
        key = f"{sample_id}/{step:04d}/{name}"
        occurrence = self._record_counts.get(key, 0)
        self._record_counts[key] = occurrence + 1
        record_id = f"{key}/{occurrence:02d}"
        self.store.append(
            record_id,
            {"value": tensor},
            kind="activation",
            sample_id=sample_id,
            split=split,
            metadata={"name": name, **metadata},
        )

    def _extract_weights(self) -> int:
        count = 0
        for name, tensor in self.adapter.iter_projection_tensors():
            if self.store.append(
                f"weight/{name}",
                {"weight": tensor},
                kind="weight",
                sample_id="teacher",
                split="all",
                metadata={"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            ):
                count += 1
        return count

    def run(self, prompts: list[PromptRecord]) -> dict[str, Any]:
        self.config.validate()
        self.adapter.load()
        peak = 0
        try:
            metadata = {
                "milestone": "M2",
                "metric_scope": "teacher internal behavior; not perceptual equivalence",
                "commit_sha": git_commit(),
                "seed": self.config.data.seed,
                "config": self.config.to_dict(),
                "teacher": self.adapter.metadata(),
                "hardware": {
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "vram_bytes": torch.cuda.get_device_properties(0).total_memory
                    if torch.cuda.is_available()
                    else 0,
                    "platform": platform.platform(),
                },
                "sample_counts": {
                    "train": sum(item.split == "train" for item in prompts),
                    "eval": sum(item.split == "eval" for item in prompts),
                },
            }
            (self.store.root / "metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
            (self.store.root / "prompts.json").write_text(
                json.dumps([item.to_dict() for item in prompts], indent=2), encoding="utf-8"
            )
            new_weights = self._extract_weights()
            self.adapter.install_capture_hooks(self._capture)
            completed = 0
            skipped = 0
            for index, prompt in enumerate(prompts):
                if self.store.sample_complete(prompt.sample_id):
                    skipped += 1
                    continue
                self._record_counts.clear()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                self.adapter.run_prompt(
                    prompt.text,
                    sample_id=prompt.sample_id,
                    split=prompt.split,
                    seed=self.config.data.seed + index,
                    frames=self.config.generation.frames,
                    height=self.config.generation.height,
                    width=self.config.generation.width,
                    steps=self.config.generation.steps,
                    guidance_scale=self.config.generation.guidance_scale,
                    max_sequence_length=self.config.generation.max_sequence_length,
                )
                if torch.cuda.is_available():
                    peak = max(peak, torch.cuda.max_memory_allocated())
                self.store.mark_sample_complete(prompt.sample_id)
                completed += 1
            result = {
                "new_weights": new_weights,
                "completed_samples": completed,
                "skipped_samples": skipped,
                "peak_vram_bytes": peak,
                "output_dir": str(self.store.root),
            }
            (self.store.root / "reports" / "extraction.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            return result
        finally:
            self.adapter.unload()
