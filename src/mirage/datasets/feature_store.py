from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class FeatureRecord:
    record_id: str
    relative_path: str
    kind: str
    sample_id: str
    split: str
    metadata: dict[str, Any]


class FeatureStore:
    """Append-only, chunked Safetensors store with atomic resumable manifests."""

    VERSION = 1

    def __init__(self, root: str | Path, flush_every: int = 64):
        if flush_every < 1:
            raise ValueError("flush_every must be positive")
        self.root = Path(root)
        self._flush_every = flush_every
        self._dirty_records = 0
        for name in ("weights", "activations", "temporal", "reports"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.Lock()
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self._manifest.get("version") != self.VERSION:
                raise ValueError("unsupported feature manifest version")
        else:
            self._manifest = {"version": self.VERSION, "records": {}, "completed_samples": []}
            self._flush()

    def _flush(self) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._manifest, indent=2, sort_keys=True), encoding="utf-8")
        for attempt in range(6):
            try:
                os.replace(temporary, self.manifest_path)
                self._dirty_records = 0
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.02 * 2**attempt)

    def has_record(self, record_id: str) -> bool:
        return record_id in self._manifest["records"]

    def sample_complete(self, sample_id: str) -> bool:
        return sample_id in self._manifest["completed_samples"]

    def mark_sample_complete(self, sample_id: str) -> None:
        with self._lock:
            if sample_id not in self._manifest["completed_samples"]:
                self._manifest["completed_samples"].append(sample_id)
                self._flush()

    def append(
        self,
        record_id: str,
        tensors: dict[str, torch.Tensor],
        *,
        kind: str,
        sample_id: str,
        split: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Persist one CPU tensor chunk; returns False when an identical record is already present."""
        if not tensors:
            raise ValueError("a feature record must contain at least one tensor")
        with self._lock:
            if self.has_record(record_id):
                return False
            folder = {"weight": "weights", "activation": "activations", "temporal": "temporal"}.get(
                kind
            )
            if folder is None:
                raise ValueError(f"unsupported feature kind: {kind}")
            safe_id = record_id.replace("/", "__").replace("\\", "__")
            relative = Path(folder) / f"{safe_id}.safetensors"
            destination = self.root / relative
            # Own the storage: teacher tensors may originate from a memory-mapped
            # checkpoint whose Safetensors handle closes immediately after capture.
            clean = {
                key: value.detach().to(device="cpu").contiguous().clone()
                for key, value in tensors.items()
            }
            temporary = destination.with_suffix(".tmp")
            save_file(clean, str(temporary))
            os.replace(temporary, destination)
            entry = FeatureRecord(
                record_id, relative.as_posix(), kind, sample_id, split, metadata or {}
            )
            self._manifest["records"][record_id] = entry.__dict__
            self._dirty_records += 1
            if self._dirty_records >= self._flush_every:
                self._flush()
            return True

    def records(
        self, *, kind: str | None = None, sample_id: str | None = None, split: str | None = None
    ) -> Iterator[FeatureRecord]:
        for raw in self._manifest["records"].values():
            record = FeatureRecord(**raw)
            if kind is not None and record.kind != kind:
                continue
            if sample_id is not None and record.sample_id != sample_id:
                continue
            if split is not None and record.split != split:
                continue
            yield record

    def load(
        self, record: FeatureRecord | str, device: str | torch.device = "cpu"
    ) -> dict[str, torch.Tensor]:
        if isinstance(record, str):
            try:
                record = FeatureRecord(**self._manifest["records"][record])
            except KeyError as error:
                raise KeyError(f"unknown feature record: {record}") from error
        return load_file(str(self.root / record.relative_path), device=str(device))
