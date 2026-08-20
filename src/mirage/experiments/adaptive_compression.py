from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..compression.adaptive import allocate_residual_ranks, contiguous_behavior_groups
from ..compression.hierarchical_fit import fit_hierarchical_shared_basis
from ..compression.shared_basis_fit import projection_family
from ..datasets import FeatureStore
from ..m2_config import M2Config
from .activation_reconstruction import run_activation_reconstruction


def regenerate_adaptive_artifact(
    config: M2Config,
    metadata_path: str | Path,
    device: str | torch.device | None = None,
) -> Path:
    """Rebuild one pruned factor tensor exactly from its retained compact metadata."""
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    store = FeatureStore(config.output_dir)
    device = torch.device(device or config.teacher.device)
    wanted = set(metadata["layer_names"])
    records = [
        record
        for record in store.records(kind="weight")
        if str(record.metadata["name"]) in wanted
    ]
    records.sort(key=lambda record: str(record.metadata["name"]))
    if [str(record.metadata["name"]) for record in records] != metadata["layer_names"]:
        raise ValueError("retained metadata does not match extracted teacher layers")
    result = fit_hierarchical_shared_basis(
        store,
        records,
        tuple(map(int, metadata["shape"])),
        groups=metadata["groups"],
        global_basis_count=int(metadata["global_basis_count"]),
        group_basis_count=int(metadata["group_basis_count"]),
        rank_per_layer=list(map(int, metadata["rank_per_layer"])),
        device=device,
        row_chunk_size=int(metadata["row_chunk_size"]),
        seed=config.data.seed,
    )
    result.metrics.update(metadata)
    artifact = metadata_path.with_suffix(".safetensors")
    result.save(artifact)
    return artifact


def run_adaptive_compression(
    config: M2Config, device: str | torch.device | None = None
) -> dict[str, Any]:
    store = FeatureStore(config.output_dir)
    device = torch.device(device or config.teacher.device)
    sensitivity_path = Path(config.output_dir) / "reports" / "sensitivity.json"
    sensitivity = (
        json.loads(sensitivity_path.read_text(encoding="utf-8"))
        if sensitivity_path.exists()
        else {}
    )
    groups_by_shape: dict[tuple[str, tuple[int, ...]], list[Any]] = defaultdict(list)
    for record in store.records(kind="weight"):
        name = str(record.metadata["name"])
        shape = tuple(map(int, record.metadata["shape"]))
        groups_by_shape[(projection_family(name), shape)].append(record)
    rows: list[dict[str, Any]] = []
    artifact_root = Path(config.output_dir) / "reports" / "adaptive_fits"
    artifact_root.mkdir(parents=True, exist_ok=True)
    recovery = config.recovery
    for (family, shape), records in sorted(groups_by_shape.items()):
        records.sort(key=lambda record: str(record.metadata["name"]))
        names = [str(record.metadata["name"]) for record in records]
        groups = contiguous_behavior_groups(
            names,
            sensitivity,
            recovery.group_count,
            minimum_group_size=recovery.minimum_group_size,
        )
        original = len(records) * shape[0] * shape[1]
        shared = (
            (recovery.global_basis_count + recovery.group_count * recovery.group_basis_count)
            * shape[0]
            * shape[1]
            + len(records) * (recovery.global_basis_count + recovery.group_basis_count)
        )
        ranks, allocation = allocate_residual_ranks(
            names,
            sensitivity,
            recovery.residual_rank_tiers,
            original_params=original,
            shared_params=shared,
            rank_cost=shape[0] + shape[1],
            target_compression_ratio=recovery.target_compression_ratio,
        )
        result = fit_hierarchical_shared_basis(
            store,
            records,
            shape,
            groups=groups,
            global_basis_count=recovery.global_basis_count,
            group_basis_count=recovery.group_basis_count,
            rank_per_layer=ranks,
            device=device,
            row_chunk_size=config.compression.streamed_row_chunk_size,
            seed=config.data.seed,
        )
        result.metrics.update(
            {
                "teacher": config.teacher.model_id,
                "family": family,
                "layer_names": names,
                "shape": list(shape),
                "allocation": allocation,
                "group_layer_names": [[names[index] for index in group] for group in groups],
            }
        )
        tier_tag = "-".join(map(str, recovery.residual_rank_tiers))
        stem = (
            f"{family.replace('.', '-')}_{shape[0]}x{shape[1]}_adaptive_"
            f"g{recovery.global_basis_count}_c{recovery.group_basis_count}_r{tier_tag}"
        )
        artifact = artifact_root / f"{stem}.safetensors"
        result.save(artifact)
        rows.append(
            {
                "family": family,
                "shape": list(shape),
                "status": "ok",
                "compression_ratio": result.metrics["compression_ratio"],
                "weight_relative_error": result.metrics["weight_relative_error"],
                "artifact": str(artifact.relative_to(config.output_dir)),
                "allocation": allocation,
                "groups": result.metrics["group_layer_names"],
            }
        )
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {"teacher": config.teacher.model_id, "recovery": config.to_dict()["recovery"], "rows": rows}
    (Path(config.output_dir) / "reports" / "adaptive_compression.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def run_adaptive_activation_sweep(
    config: M2Config,
    device: str | torch.device | None = None,
    *,
    behavior_fit: bool = False,
) -> dict[str, Any]:
    root = Path(config.output_dir)
    source = json.loads((root / "reports" / "adaptive_compression.json").read_text())
    rows = []
    for row in source["rows"]:
        report = run_activation_reconstruction(
            config, row["artifact"], device=device, behavior_fit=behavior_fit
        )
        rows.append(
            {
                "family": row["family"],
                "artifact": row["artifact"],
                "compression_ratio": row["compression_ratio"],
                "validation_mean_relative_error": report["validation_mean_relative_error"],
                "validation_mean_cosine": report["validation_mean_cosine"],
                "behavior_fit": behavior_fit,
                "evaluated_artifact": report["evaluated_artifact"],
            }
        )
    result = {"teacher": config.teacher.model_id, "rows": rows}
    (root / "reports" / "adaptive_activation_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
