from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..compression.activation_fit import activation_metrics, optimize_for_activations
from ..compression.factorization import FactorizationResult
from ..datasets import FeatureStore
from ..m2_config import M2Config


def _inputs_by_projection(store: FeatureStore, split: str) -> dict[str, list[torch.Tensor]]:
    values: dict[str, list[torch.Tensor]] = defaultdict(list)
    for record in store.records(kind="activation", split=split):
        if record.metadata.get("hook") != "projection_input":
            continue
        name = str(record.metadata["projection_name"])
        values[name].append(store.load(record)["value"])
    return values


def _concatenate(records: list[torch.Tensor], width: int, max_tokens: int = 4096) -> torch.Tensor:
    compatible = [
        tensor.reshape(-1, tensor.shape[-1]) for tensor in records if tensor.shape[-1] == width
    ]
    if not compatible:
        raise ValueError(f"no captured projection inputs with width {width}")
    return torch.cat(compatible, dim=0)[:max_tokens]


def run_activation_reconstruction(
    config: M2Config,
    artifact: str | Path,
    *,
    behavior_fit: bool = False,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    store = FeatureStore(config.output_dir)
    device = torch.device(device or config.teacher.device)
    artifact = Path(artifact)
    if not artifact.is_absolute():
        artifact = Path(config.output_dir) / artifact
    fit = FactorizationResult.load(artifact, device=device)
    layer_names = list(fit.metrics.get("layer_names", []))
    if len(layer_names) != fit.alpha.shape[0]:
        raise ValueError("factorization artifact lacks an exact layer_names mapping")
    weights = []
    for name in layer_names:
        weights.append(store.load(f"weight/{name}", device=device)["weight"].float())
    source = torch.stack(weights)
    train_records = _inputs_by_projection(store, "train")
    eval_records = _inputs_by_projection(store, "eval")
    train_inputs = [
        _concatenate(train_records.get(name, []), source[i].shape[-1]).to(device)
        for i, name in enumerate(layer_names)
    ]
    eval_inputs = [
        _concatenate(eval_records.get(name, []), source[i].shape[-1]).to(device)
        for i, name in enumerate(layer_names)
    ]
    behavior_report = None
    if behavior_fit:
        fit, behavior_report = optimize_for_activations(
            fit,
            source,
            train_inputs,
            eval_inputs,
            steps=config.compression.activation_fit_steps,
            learning_rate=config.compression.learning_rate,
            lambda_weight=config.compression.lambda_weight,
            lambda_activation=config.compression.lambda_activation,
        )
        fitted_path = artifact.with_name(f"{artifact.stem}_activation_fit.safetensors")
        fit.save(fitted_path)
    reconstructed = fit.reconstruct()
    train = [
        activation_metrics(train_inputs[i], source[i], reconstructed[i]).to_dict()
        for i in range(len(layer_names))
    ]
    validation = [
        activation_metrics(eval_inputs[i], source[i], reconstructed[i]).to_dict()
        for i in range(len(layer_names))
    ]
    report = {
        "artifact": str(artifact),
        "teacher": config.teacher.model_id,
        "layer_names": layer_names,
        "behavior_fit": behavior_report,
        "train": train,
        "validation": validation,
        "train_mean_relative_error": sum(x["relative_activation_error"] for x in train)
        / len(train),
        "validation_mean_relative_error": sum(x["relative_activation_error"] for x in validation)
        / len(validation),
        "train_mean_cosine": sum(x["cosine_similarity"] for x in train) / len(train),
        "validation_mean_cosine": sum(x["cosine_similarity"] for x in validation) / len(validation),
        "scope_warning": "Local projection fidelity is not full-generation perceptual equivalence.",
    }
    target = Path(config.output_dir) / "reports" / f"activation_{artifact.stem}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
