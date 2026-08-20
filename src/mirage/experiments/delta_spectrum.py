from __future__ import annotations

import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from ..datasets import FeatureStore
from ..m2_config import M2Config
from .feature_sequences import load_block_sequences


def delta_spectrum(delta: torch.Tensor, ranks: list[int]) -> dict[str, Any]:
    matrix = delta.float().reshape(-1, delta.shape[-1])
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    total = energy.sum().clamp_min(1e-12)
    cumulative = energy.cumsum(0) / total
    rows = {}
    for rank in ranks:
        effective = min(rank, len(singular))
        explained = cumulative[effective - 1].item()
        rows[str(rank)] = {
            "effective_rank": effective,
            "factor_compression_ratio": matrix.numel()
            / (effective * (matrix.shape[0] + matrix.shape[1])),
            "explained_energy": explained,
            "relative_reconstruction_error": max(0.0, 1.0 - explained) ** 0.5,
        }
    return {"matrix_shape": list(matrix.shape), "ranks": rows}


def run_delta_spectrum(config: M2Config) -> dict[str, Any]:
    sequences = load_block_sequences(FeatureStore(config.output_dir))
    by_split: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for values in sequences.values():
        for previous, current in pairwise(values):
            delta = current.residual.float() - previous.residual.float()
            by_split[current.split][current.block_index].append(
                {
                    "sample_id": current.sample_id,
                    "step_index": current.step_index,
                    **delta_spectrum(delta, config.temporal.delta_ranks),
                }
            )
    summaries: dict[str, Any] = {}
    for split, layers in by_split.items():
        rank_summary = {}
        all_rows = [row for rows in layers.values() for row in rows]
        for rank in config.temporal.delta_ranks:
            errors = [row["ranks"][str(rank)]["relative_reconstruction_error"] for row in all_rows]
            energies = [row["ranks"][str(rank)]["explained_energy"] for row in all_rows]
            compression = [
                row["ranks"][str(rank)]["factor_compression_ratio"] for row in all_rows
            ]
            accepted = sum(error <= config.temporal.delta_local_error for error in errors)
            rank_summary[str(rank)] = {
                "mean_explained_energy": sum(energies) / len(energies),
                "mean_relative_reconstruction_error": sum(errors) / len(errors),
                "accepted_local_error_coverage": accepted / len(errors),
                "mean_factor_compression_ratio": sum(compression) / len(compression),
                "transition_count": len(errors),
            }
        summaries[split] = rank_summary
    evaluation = summaries.get("eval")
    if not evaluation:
        raise ValueError("delta spectrum requires held-out evaluation sequences")
    eligible_ranks = [
        rank
        for rank in config.temporal.delta_ranks
        if evaluation[str(rank)]["mean_factor_compression_ratio"]
        >= config.temporal.delta_min_compression_ratio
    ]
    if not eligible_ranks:
        raise ValueError("no delta rank meets the configured factor compression ratio")
    best_rank = max(
        eligible_ranks,
        key=lambda rank: (
            evaluation[str(rank)]["accepted_local_error_coverage"],
            -evaluation[str(rank)]["mean_relative_reconstruction_error"],
        ),
    )
    best = evaluation[str(best_rank)]
    report = {
        "teacher": config.teacher.model_id,
        "metric": "SVD spectrum of delta_R across sampled video tokens and channels",
        "summaries": summaries,
        "layers": {
            split: {str(layer): rows for layer, rows in layers.items()}
            for split, layers in by_split.items()
        },
        "decision": {
            "tested_rank": best_rank,
            "accepted_local_error": config.temporal.delta_local_error,
            "minimum_factor_compression_ratio": config.temporal.delta_min_compression_ratio,
            "eligible_ranks": eligible_ranks,
            "held_out_useful_coverage": best["accepted_local_error_coverage"],
            "low_rank_delta_promising": (
                best["accepted_local_error_coverage"]
                >= config.temporal.delta_useful_coverage
            ),
        },
        "scope_warning": (
            "This is an oracle low-rank spectrum test. It does not claim that a causal delta "
            "adapter can predict the measured subspace."
        ),
    }
    target = Path(config.output_dir) / "temporal" / "delta_spectrum.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
