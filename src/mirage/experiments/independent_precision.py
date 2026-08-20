from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..compression.activation_fit import activation_metrics
from ..datasets import FeatureRecord, FeatureStore
from ..m2_config import M2Config
from .sparse_residual import _projection_inputs


def quantize_int8_rows(weight: torch.Tensor) -> tuple[torch.Tensor, int]:
    scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
    quantized = (weight.float() / scale).round().clamp(-127, 127).to(torch.int8)
    reconstructed = quantized.float() * scale
    return reconstructed, quantized.numel() + scale.numel() * 2


def quantize_fp8_rows(weight: torch.Tensor) -> tuple[torch.Tensor, int]:
    maximum = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / maximum
    quantized = (weight.float() / scale).to(torch.float8_e4m3fn)
    reconstructed = quantized.float() * scale
    return reconstructed, quantized.numel() + scale.numel() * 2


def quantize_int4_groups(weight: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, int]:
    """Symmetric signed INT4 with one BF16 scale per contiguous input group."""
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(f"INT4 group size {group_size} does not divide {columns}")
    grouped = weight.float().reshape(rows, columns // group_size, group_size)
    scale = grouped.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / 7
    quantized = (grouped / scale).round().clamp(-7, 7).to(torch.int8)
    reconstructed = (quantized.float() * scale).reshape_as(weight)
    packed_bytes = (quantized.numel() + 1) // 2
    return reconstructed, packed_bytes + scale.numel() * 2


def _records_by_family(store: FeatureStore) -> dict[str, list[FeatureRecord]]:
    result: dict[str, list[FeatureRecord]] = defaultdict(list)
    for record in store.records(kind="weight"):
        name = str(record.metadata["name"])
        family = ".".join(name.split(".")[2:])
        family = {"ff.in": "ff-in", "ff.out": "ff-out"}.get(family, family)
        result[family].append(record)
    return result


def run_independent_precision_study(
    config: M2Config, device: str | torch.device | None = None
) -> dict[str, Any]:
    root = Path(config.output_dir)
    store = FeatureStore(root)
    device = torch.device(device or config.teacher.device)
    eval_records: dict[str, list[FeatureRecord]] = defaultdict(list)
    for record in store.records(kind="activation", split="eval"):
        if record.metadata.get("hook") == "projection_input":
            eval_records[str(record.metadata["projection_name"])].append(record)
    families = {}
    for family, records in sorted(_records_by_family(store).items()):
        records.sort(key=lambda record: str(record.metadata["name"]))
        methods: dict[str, list[dict[str, Any]]] = {
            "INT4_GROUP64": [],
            "INT8_ROW": [],
            "FP8_ROW": [],
        }
        for record in records:
            name = str(record.metadata["name"])
            source = store.load(record, device=device)["weight"].float()
            inputs = _projection_inputs(store, eval_records[name], source.shape[1]).to(device)
            for method, function in (
                ("INT4_GROUP64", quantize_int4_groups),
                ("INT8_ROW", quantize_int8_rows),
                ("FP8_ROW", quantize_fp8_rows),
            ):
                reconstructed, stored_bytes = function(source)
                metrics = activation_metrics(inputs, source, reconstructed).to_dict()
                methods[method].append(
                    {
                        "layer": name,
                        "stored_bytes": stored_bytes,
                        "activation_error": metrics["relative_activation_error"],
                        "activation_cosine": metrics["cosine_similarity"],
                    }
                )
                del reconstructed
            del source, inputs
        original_bytes = sum(math.prod(record.metadata["shape"]) * 2 for record in records)
        summary = {}
        for method, layers in methods.items():
            stored = sum(int(layer["stored_bytes"]) for layer in layers)
            summary[method] = {
                "stored_bytes": stored,
                "compression_ratio": original_bytes / stored,
                "mean_activation_error": sum(layer["activation_error"] for layer in layers)
                / len(layers),
                "mean_activation_cosine": sum(layer["activation_cosine"] for layer in layers)
                / len(layers),
            }
        families[family] = {"summary": summary, "layers": methods}
    report = {
        "teacher": config.teacher.model_id,
        "held_out_only": True,
        "scale_storage": "BF16 per output row for 8-bit; BF16 per 64 input values for INT4",
        "families": families,
    }
    (root / "reports" / "independent_precision.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
