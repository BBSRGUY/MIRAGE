from __future__ import annotations

import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from ..datasets import FeatureStore
from ..m2_config import M2Config
from ..temporal.cache_analysis import (
    cache_threshold_sweep,
    execution_policy,
    reuse_error,
    summarize_execution,
)
from ..temporal.predictor import fit_predictor
from .feature_sequences import BlockStepFeatures, load_block_sequences


def _pairs(sequences: dict[tuple[str, int], list[BlockStepFeatures]], split: str | None = None):
    for (_sample, block), steps in sequences.items():
        for previous, current in pairwise(steps):
            if split is None or current.split == split:
                yield block, previous, current


def run_cache_analysis(config: M2Config) -> dict[str, Any]:
    sequences = load_block_sequences(FeatureStore(config.output_dir))
    by_block: dict[int, list[dict[str, float]]] = defaultdict(list)
    all_errors = []
    for block, previous, current in _pairs(sequences):
        metrics = reuse_error(current.block_input, current.block_output, previous.residual)
        metrics.update({"step_index": current.step_index, "timestep": current.timestep})
        by_block[block].append(metrics)
        all_errors.append(metrics)
    report = {
        "teacher": config.teacher.model_id,
        "threshold_sweep": cache_threshold_sweep(all_errors, config.temporal.reuse_thresholds),
        "layers": {str(key): value for key, value in by_block.items()},
        "scope_warning": "Cache replay is a local teacher-block study, not full-video equivalence.",
    }
    target = Path(config.output_dir) / "temporal" / "cache_analysis.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _predictor_sample(
    previous: BlockStepFeatures, current: BlockStepFeatures, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    delta = current.block_input.float() - previous.block_input.float()
    timestep = current.timestep / max(abs(current.timestep), 1.0)
    return (
        delta.to(device),
        previous.residual.float().to(device),
        current.residual.float().to(device),
        timestep,
    )


def run_predictor_fit(config: M2Config, device: str | torch.device | None = None) -> dict[str, Any]:
    device = torch.device(device or config.teacher.device)
    sequences = load_block_sequences(FeatureStore(config.output_dir))
    block_ids = sorted({block for _sample, block in sequences})
    reports: dict[str, Any] = {}
    all_decisions: list[str] = []
    all_prediction_errors: list[float] = []
    predictor_root = Path(config.output_dir) / "temporal" / "predictors"
    predictor_root.mkdir(parents=True, exist_ok=True)
    total_parameters = 0
    for block in block_ids:
        train_samples = [
            _predictor_sample(previous, current, device)
            for candidate, previous, current in _pairs(sequences, "train")
            if candidate == block
        ]
        validation_pairs = [
            (previous, current)
            for candidate, previous, current in _pairs(sequences, "eval")
            if candidate == block
        ]
        validation_samples = [
            _predictor_sample(previous, current, device) for previous, current in validation_pairs
        ]
        if not train_samples or not validation_samples:
            reports[str(block)] = {"status": "skipped_missing_split"}
            continue
        width = train_samples[0][0].shape[-1]
        fitted = fit_predictor(
            width,
            train_samples,
            validation_samples,
            steps=config.temporal.predictor_steps,
            learning_rate=config.temporal.predictor_learning_rate,
            seed=config.data.seed + block,
        )
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in fitted.model.state_dict().items()
            },
            str(predictor_root / f"block_{block:02d}.safetensors"),
        )
        total_parameters += fitted.parameter_count
        decisions, prediction_errors = [], []
        fitted.model.eval()
        with torch.no_grad():
            for (previous, current), sample in zip(validation_pairs, validation_samples):
                delta, prior, target, timestep = sample
                prediction = prior + fitted.model(delta, torch.tensor(timestep, device=device))
                prediction_error = (prediction - target).norm() / target.norm().clamp_min(1e-12)
                reuse = reuse_error(current.block_input, current.block_output, previous.residual)
                decision = execution_policy(
                    reuse["residual_relative_error"],
                    prediction_error.item(),
                    config.temporal.reuse_threshold,
                    config.temporal.predict_threshold,
                )
                decisions.append(decision)
                prediction_errors.append(prediction_error.item())
        all_decisions.extend(decisions)
        all_prediction_errors.extend(prediction_errors)
        reports[str(block)] = {
            "status": "ok",
            "train_relative_error": fitted.train_relative_error,
            "validation_relative_error": fitted.validation_relative_error,
            "validation_cosine": fitted.validation_cosine,
            "predictor_parameter_count": fitted.parameter_count,
            "mean_policy_prediction_error": sum(prediction_errors) / len(prediction_errors),
            **summarize_execution(decisions, fitted.parameter_count),
        }
    summary = summarize_execution(all_decisions, total_parameters)
    summary["mean_prediction_error"] = sum(all_prediction_errors) / len(all_prediction_errors)
    report = {
        "teacher": config.teacher.model_id,
        "layers": reports,
        "summary": summary,
        "scope_warning": "Predictor coverage measures held-out local residual behavior only.",
    }
    (Path(config.output_dir) / "temporal" / "predictor_fit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
