from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..compression.factorization import FitOptions, fit_shared_basis
from ..compression.shared_basis_fit import compatible_groups, load_weight_families
from ..datasets import FeatureStore
from ..m2_config import M2Config

SUMMARY_FIELDS = [
    "teacher",
    "projection_family",
    "shape",
    "basis_count",
    "rank",
    "status",
    "original_params",
    "compressed_params",
    "compression_ratio",
    "weight_relative_error",
    "median_layer_error",
    "p95_layer_error",
    "max_layer_error",
    "fit_seconds",
    "artifact",
]


def run_basis_sweep(config: M2Config, device: str | torch.device | None = None) -> dict[str, Any]:
    store = FeatureStore(config.output_dir)
    device = torch.device(device or config.teacher.device)
    families = load_weight_families(store, device="cpu")
    rows: list[dict[str, Any]] = []
    artifact_root = Path(config.output_dir) / "reports" / "basis_fits"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for family, named_weights in families.items():
        for shape, group in compatible_groups(named_weights):
            names, tensors = zip(*sorted(group))
            layer_count = len(tensors)
            weights = torch.stack([tensor.float() for tensor in tensors]).to(device)
            for basis_count in config.compression.basis_counts:
                for rank in config.compression.ranks:
                    row = {
                        "teacher": config.teacher.model_id,
                        "projection_family": family,
                        "shape": "x".join(map(str, shape)),
                        "basis_count": basis_count,
                        "rank": rank,
                    }
                    if basis_count > layer_count or rank > min(shape):
                        rows.append({**row, "status": "skipped_invalid"})
                        continue
                    options = FitOptions(
                        basis_count=basis_count,
                        rank=rank,
                        initialization=config.compression.initialization,
                        optimization_steps=config.compression.fit_steps,
                        learning_rate=config.compression.learning_rate,
                        alpha_sparsity=config.compression.alpha_sparsity,
                        residual_magnitude=config.compression.residual_magnitude,
                        basis_orthogonality=config.compression.basis_orthogonality,
                        seed=config.data.seed,
                    )
                    result = fit_shared_basis(weights, options)
                    stem = f"{family.replace('.', '-')}_{row['shape']}_k{basis_count}_r{rank}"
                    artifact = artifact_root / f"{stem}.safetensors"
                    result.metrics.update(
                        {
                            "teacher": config.teacher.model_id,
                            "family": family,
                            "layer_names": list(names),
                            "shape": list(shape),
                        }
                    )
                    result.save(artifact)
                    rows.append(
                        {
                            **row,
                            "status": "ok",
                            **{
                                key: result.metrics[key]
                                for key in SUMMARY_FIELDS
                                if key in result.metrics
                            },
                            "artifact": str(artifact.relative_to(config.output_dir)),
                        }
                    )
            del weights
            if device.type == "cuda":
                torch.cuda.empty_cache()
    reports = Path(config.output_dir) / "reports"
    (reports / "basis_sweep.json").write_text(
        json.dumps({"config": asdict(config.compression), "rows": rows}, indent=2), encoding="utf-8"
    )
    with (reports / "basis_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "rows": rows,
        "completed": sum(row.get("status") == "ok" for row in rows),
        "skipped": sum(row.get("status") != "ok" for row in rows),
    }
