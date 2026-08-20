from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from ..datasets import FeatureStore
from ..m3_config import M3Config
from ..training.losses import behavior_signature


def build_m3_teacher_features(config: M3Config, m2_root: str | Path) -> dict[str, Any]:
    """Convert M2 activation chunks into small offline behavior signatures."""
    source = FeatureStore(m2_root)
    output = Path(config.data.teacher_features or Path(config.training.output_dir) / "teacher_features")
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for record in source.records(kind="activation"):
        if record.metadata.get("hook") in {
            "block_output",
            "block_residual",
            "attention_output",
            "projection_output",
        }:
            grouped[(record.sample_id, record.split)].append(record)
    manifest = []
    for (sample_id, split), records in sorted(grouped.items()):
        signatures = []
        for record in records:
            values = source.load(record)
            tensor = next(iter(values.values()))
            signatures.append(behavior_signature(tensor.reshape(1, -1, tensor.shape[-1]))[0])
        signature = torch.stack(signatures).mean(0) if signatures else torch.zeros(4)
        destination = output / f"{sample_id}.safetensors"
        save_file({"signature": signature.contiguous()}, str(destination))
        manifest.append(
            {
                "sample_id": sample_id,
                "split": split,
                "teacher_feature": destination.name,
                "records_aggregated": len(records),
            }
        )
    report = {
        "format": "mirage_m3_teacher_signature_v1",
        "source": str(m2_root),
        "samples": manifest,
        "teacher_required_at_training_or_runtime": False,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
