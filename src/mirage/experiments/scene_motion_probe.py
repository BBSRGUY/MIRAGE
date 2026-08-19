from __future__ import annotations

import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from ..datasets import FeatureStore
from ..m2_config import M2Config
from ..temporal.drift import drift_metrics
from ..temporal.scene_motion_analysis import analyze_scene_motion, temporal_mean_decomposition


def _reshape_temporal(value: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    frames = int(metadata.get("latent_num_frames", 0))
    if frames < 1:
        raise ValueError("captured LTX activation lacks latent_num_frames metadata")
    value = value.float()
    if value.ndim != 3:
        raise ValueError(f"expected captured [batch,tokens,channels], found {tuple(value.shape)}")
    indices = metadata.get("token_indices")
    if indices is None:
        if value.shape[1] % frames:
            raise ValueError("captured token count is not divisible by latent frame count")
        return value.view(value.shape[0], frames, value.shape[1] // frames, value.shape[2])
    original_tokens = int(metadata["original_shape"][-2])
    if original_tokens % frames:
        raise ValueError("original token count is not divisible by latent frame count")
    spatial = original_tokens // frames
    grouped: list[list[int]] = [[] for _ in range(frames)]
    for local_index, original_index in enumerate(indices):
        grouped[min(int(original_index) // spatial, frames - 1)].append(local_index)
    keep = min(map(len, grouped))
    if keep < 1:
        raise ValueError("token sampling omitted at least one latent frame")
    return torch.stack([value[:, positions[:keep]] for positions in grouped], dim=1)


def run_scene_motion_probe(config: M2Config) -> dict[str, Any]:
    store = FeatureStore(config.output_dir)
    per_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    scenes: dict[tuple[str, int], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for record in store.records(kind="activation"):
        if record.metadata.get("hook") != "block_output":
            continue
        block = int(record.metadata["block_index"])
        value = _reshape_temporal(store.load(record)["value"], record.metadata)
        analysis = analyze_scene_motion(
            value, config.scene_motion.pca_ranks, config.scene_motion.lowpass_alpha
        )
        per_layer[block].append(analysis)
        scene, _ = temporal_mean_decomposition(value)
        scenes[(record.sample_id, block)].append((int(record.metadata["step_index"]), scene[:, 0]))
    if not per_layer:
        raise ValueError("no block outputs available for scene/motion analysis")
    layers: dict[str, Any] = {}
    for block, rows in per_layer.items():
        pca: dict[str, Any] = {}
        for rank in map(str, config.scene_motion.pca_ranks):
            valid = [
                row["temporal_pca"][rank]
                for row in rows
                if row["temporal_pca"][rank]["status"] == "ok"
            ]
            pca[rank] = (
                {"status": "skipped_invalid"}
                if not valid
                else {
                    "status": "ok",
                    **{
                        key: sum(item[key] for item in valid) / len(valid)
                        for key in valid[0]
                        if key != "status"
                    },
                }
            )
        mean_rows = [row["temporal_mean"] for row in rows]
        layers[str(block)] = {
            "temporal_mean": {
                key: sum(row[key] for row in mean_rows) / len(mean_rows) for key in mean_rows[0]
            },
            "temporal_pca": pca,
            "samples": len(rows),
        }
    stability = []
    for values in scenes.values():
        values.sort(key=lambda item: item[0])
        stability.extend(drift_metrics(a[1], b[1]).normalized_drift for a, b in pairwise(values))
    rank_one = [
        layer["temporal_pca"]["1"]["explained_energy"]
        for layer in layers.values()
        if layer["temporal_pca"]["1"]["status"] == "ok"
    ]
    report = {
        "teacher": config.teacher.model_id,
        "layers": layers,
        "summary": {
            "mean_rank1_explained_energy": sum(rank_one) / len(rank_one),
            "mean_scene_stability_drift": sum(stability) / len(stability) if stability else None,
        },
        "scope_warning": "Low-rank feature structure does not establish an optimal generative representation.",
    }
    (Path(config.output_dir) / "temporal" / "scene_motion.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
