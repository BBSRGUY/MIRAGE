from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def reuse_error(
    current_input: torch.Tensor, current_output: torch.Tensor, previous_residual: torch.Tensor
) -> dict[str, float]:
    """Replay R[t-1] at X[t] and compare to the true local block output."""
    if (
        current_input.shape != current_output.shape
        or current_input.shape != previous_residual.shape
    ):
        raise ValueError("cache replay tensors must have identical shapes")
    true_residual = current_output.float() - current_input.float()
    reused_output = current_input.float() + previous_residual.float()
    difference = reused_output - current_output.float()
    return {
        "reuse_output_relative_error": (
            difference.norm() / current_output.float().norm().clamp_min(1e-12)
        ).item(),
        "residual_relative_error": (
            (previous_residual.float() - true_residual).norm()
            / true_residual.norm().clamp_min(1e-12)
        ).item(),
        "residual_cosine_similarity": F.cosine_similarity(
            previous_residual.float().flatten(), true_residual.flatten(), dim=0
        ).item(),
        "downstream_local_reconstruction_error": difference.square().mean().sqrt().item(),
    }


def cache_threshold_sweep(
    errors: list[dict[str, float]], thresholds: list[float]
) -> list[dict[str, float]]:
    if not errors:
        raise ValueError("cache analysis requires at least one adjacent-step pair")
    rows = []
    for threshold in thresholds:
        hits = sum(item["reuse_output_relative_error"] <= threshold for item in errors)
        rate = hits / len(errors)
        rows.append(
            {
                "threshold": threshold,
                "eligible": hits,
                "total": len(errors),
                "cache_hit_rate": rate,
                "expected_flop_reduction": rate,
            }
        )
    return rows


def execution_policy(
    reuse_error_value: float,
    prediction_error_value: float,
    reuse_threshold: float,
    predict_threshold: float,
) -> str:
    if reuse_threshold > predict_threshold:
        raise ValueError("reuse_threshold must not exceed predict_threshold")
    if reuse_error_value <= reuse_threshold:
        return "reuse"
    if prediction_error_value <= predict_threshold:
        return "predict"
    return "execute"


def summarize_execution(
    decisions: list[str],
    predictor_params: int,
    full_block_flops: float = 1.0,
    predictor_flops: float = 0.02,
) -> dict[str, Any]:
    if not decisions:
        raise ValueError("execution summary requires decisions")
    counts = {name: decisions.count(name) for name in ("reuse", "predict", "execute")}
    total = len(decisions)
    effective = counts["execute"] * full_block_flops + counts["predict"] * predictor_flops
    return {
        **{f"{name}_percentage": counts[name] / total for name in counts},
        "predictor_parameter_count": predictor_params,
        "theoretical_flop_reduction": 1.0 - effective / (total * full_block_flops),
        "decision_count": total,
    }
