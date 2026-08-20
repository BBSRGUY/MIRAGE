from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..compression.activation_fit import activation_metrics
from ..compression.factorization import load_factorization
from ..datasets import FeatureRecord, FeatureStore
from ..m2_config import M2Config


def tile_sparse_residual(
    residual: torch.Tensor, tile_size: int, density: float
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if residual.shape[0] % tile_size or residual.shape[1] % tile_size:
        raise ValueError("matrix dimensions must be divisible by tile_size")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    out_tiles = residual.shape[0] // tile_size
    in_tiles = residual.shape[1] // tile_size
    tiles = residual.view(out_tiles, tile_size, in_tiles, tile_size).permute(0, 2, 1, 3)
    energy = tiles.square().sum(dim=(-1, -2))
    count = max(1, math.ceil(energy.numel() * density))
    selected = torch.topk(energy.flatten(), count).indices
    mask = torch.zeros(energy.numel(), device=residual.device, dtype=torch.bool)
    mask[selected] = True
    mask = mask.view_as(energy)
    sparse_tiles = tiles * mask[:, :, None, None]
    sparse = sparse_tiles.permute(0, 2, 1, 3).reshape_as(residual)
    captured = sparse.square().sum() / residual.square().sum().clamp_min(1e-12)
    return sparse, {
        "selected_tiles": count,
        "total_tiles": energy.numel(),
        "actual_density": count / energy.numel(),
        "captured_residual_energy": captured.item(),
        "stored_values": count * tile_size * tile_size,
        "index_bytes": count * 4,
    }


def _records_by_projection(
    store: FeatureStore, split: str
) -> dict[str, list[FeatureRecord]]:
    result: dict[str, list[FeatureRecord]] = defaultdict(list)
    for record in store.records(kind="activation", split=split):
        if record.metadata.get("hook") == "projection_input":
            result[str(record.metadata["projection_name"])].append(record)
    return result


def _projection_inputs(
    store: FeatureStore, records: list[FeatureRecord], width: int
) -> torch.Tensor:
    values = [
        store.load(record)["value"].reshape(-1, width)
        for record in records
        if int(record.metadata["original_shape"][-1]) == width
    ]
    if not values:
        raise ValueError(f"no held-out projection inputs with width {width}")
    return torch.cat(values)[:4096]


def run_sparse_residual_study(
    config: M2Config, device: str | torch.device | None = None
) -> dict[str, Any]:
    root = Path(config.output_dir)
    store = FeatureStore(root)
    device = torch.device(device or config.teacher.device)
    adaptive = json.loads((root / "reports" / "adaptive_compression.json").read_text())
    eval_records = _records_by_projection(store, "eval")
    families: dict[str, Any] = {}
    for row in adaptive["rows"]:
        source_artifact = root / row["artifact"]
        artifact = source_artifact.with_name(
            f"{source_artifact.stem}_activation_fit.safetensors"
        )
        fit = load_factorization(artifact, device=device)
        names = list(fit.metrics["layer_names"])
        shape = tuple(map(int, fit.metrics["shape"]))
        per_density: dict[str, list[dict[str, float | int]]] = {
            str(density): [] for density in config.recovery.sparse_tile_densities
        }
        for index, name in enumerate(names):
            source = store.load(f"weight/{name}", device=device)["weight"].float()
            base = fit.reconstruct_layer(index)
            residual = source - base
            inputs = _projection_inputs(store, eval_records[name], shape[1]).to(device)
            for density in config.recovery.sparse_tile_densities:
                sparse, sparse_metrics = tile_sparse_residual(
                    residual, config.recovery.sparse_tile_size, density
                )
                metrics = activation_metrics(inputs, source, base + sparse).to_dict()
                per_density[str(density)].append(
                    {
                        "layer": name,
                        **sparse_metrics,
                        "activation_error": metrics["relative_activation_error"],
                        "activation_cosine": metrics["cosine_similarity"],
                    }
                )
                del sparse
            del source, base, residual, inputs
        density_summary = {}
        original_bytes = len(names) * math.prod(shape) * 2
        base_bytes = int(fit.metrics["compressed_params"]) * 2
        for density, layers in per_density.items():
            sparse_bytes = sum(
                int(layer["stored_values"]) * 2 + int(layer["index_bytes"])
                for layer in layers
            )
            density_summary[density] = {
                "mean_captured_residual_energy": sum(
                    float(layer["captured_residual_energy"]) for layer in layers
                )
                / len(layers),
                "mean_activation_error": sum(float(layer["activation_error"]) for layer in layers)
                / len(layers),
                "mean_activation_cosine": sum(
                    float(layer["activation_cosine"]) for layer in layers
                )
                / len(layers),
                "family_compression_ratio": original_bytes / (base_bytes + sparse_bytes),
                "sparse_bytes": sparse_bytes,
            }
        families[row["family"]] = {
            "artifact": str(artifact),
            "tile_size": config.recovery.sparse_tile_size,
            "summary": density_summary,
            "layers": per_density,
        }
        del fit
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "teacher": config.teacher.model_id,
        "selection": "top tile Frobenius energy from training-independent teacher weights",
        "families": families,
        "held_out_only": True,
    }
    (root / "reports" / "sparse_residual.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
