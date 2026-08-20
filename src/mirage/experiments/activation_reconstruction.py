from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..compression.activation_fit import activation_metrics, optimize_for_activations
from ..compression.factorization import load_factorization
from ..datasets import FeatureStore
from ..m2_config import M2Config


def _inputs_by_projection(
    store: FeatureStore, split: str, names: set[str]
) -> dict[str, list[torch.Tensor]]:
    values: dict[str, list[torch.Tensor]] = defaultdict(list)
    for record in store.records(kind="activation", split=split):
        if record.metadata.get("hook") != "projection_input":
            continue
        name = str(record.metadata["projection_name"])
        if name not in names:
            continue
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
    fit = load_factorization(artifact, device=device)
    layer_names = list(fit.metrics.get("layer_names", []))
    if len(layer_names) != fit.layer_count:
        raise ValueError("factorization artifact lacks an exact layer_names mapping")
    wanted = set(layer_names)
    train_records = _inputs_by_projection(store, "train", wanted)
    eval_records = _inputs_by_projection(store, "eval", wanted)
    train_inputs = [
        _concatenate(
            train_records.get(name, []), int(fit.metrics["shape"][-1])
        )
        for name in layer_names
    ]
    eval_inputs = [
        _concatenate(eval_records.get(name, []), int(fit.metrics["shape"][-1]))
        for name in layer_names
    ]
    behavior_report = None
    evaluated_artifact = artifact
    if behavior_fit:
        if fit.metrics.get("format") == "hierarchical_shared_basis_v1":
            layer_fits = []
            for index, name in enumerate(layer_names):
                source_layer = store.load(f"weight/{name}", device=device)["weight"].float()
                layer_fits.append(
                    fit.fit_activation_residual(
                        index,
                        source_layer,
                        train_inputs[index].to(device),
                        ridge=config.recovery.activation_residual_ridge,
                        seed=config.data.seed + index,
                    )
                )
                del source_layer
            behavior_report = {
                "method": "closed_form_activation_metric_residual_refit",
                "ridge": config.recovery.activation_residual_ridge,
                "layers": layer_fits,
                "mean_training_output_error": sum(
                    row["training_output_error"] for row in layer_fits
                )
                / len(layer_fits),
            }
        else:
            source_bytes = fit.layer_count * math.prod(fit.metrics["shape"]) * 4
            if source_bytes * 3 > config.compression.max_fit_vram_gb * 2**30:
                raise ValueError("behavior fitting requires a streamed optimizer for this family")
            source = torch.stack(
                [
                    store.load(f"weight/{name}", device=device)["weight"].float()
                    for name in layer_names
                ]
            )
            fit, behavior_report = optimize_for_activations(
                fit,
                source,
                [value.to(device) for value in train_inputs],
                [value.to(device) for value in eval_inputs],
                steps=config.compression.activation_fit_steps,
                learning_rate=config.compression.learning_rate,
                lambda_weight=config.compression.lambda_weight,
                lambda_activation=config.compression.lambda_activation,
            )
        fitted_path = artifact.with_name(f"{artifact.stem}_activation_fit.safetensors")
        fit.save(fitted_path)
        evaluated_artifact = fitted_path
    train, validation = [], []
    for i, name in enumerate(layer_names):
        source_layer = store.load(f"weight/{name}", device=device)["weight"].float()
        reconstructed_layer = fit.reconstruct_layer(i)
        train.append(
            activation_metrics(
                train_inputs[i].to(device), source_layer, reconstructed_layer
            ).to_dict()
        )
        validation.append(
            activation_metrics(
                eval_inputs[i].to(device), source_layer, reconstructed_layer
            ).to_dict()
        )
        del source_layer, reconstructed_layer
    report = {
        "artifact": str(artifact),
        "evaluated_artifact": str(evaluated_artifact),
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


def run_activation_sweep(
    config: M2Config, device: str | torch.device | None = None
) -> dict[str, Any]:
    """Evaluate every successful basis candidate, resuming completed reports."""

    root = Path(config.output_dir)
    sweep_path = root / "reports" / "basis_sweep.json"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"basis sweep report is missing: {sweep_path}")
    rows = json.loads(sweep_path.read_text(encoding="utf-8"))["rows"]
    completed, skipped = [], 0
    started = time.perf_counter()
    for row in rows:
        if row.get("status") != "ok":
            continue
        artifact = str(row["artifact"])
        target = root / "reports" / f"activation_{Path(artifact).stem}.json"
        if target.exists():
            skipped += 1
            continue
        report = run_activation_reconstruction(config, artifact, device=device)
        completed.append(
            {
                "artifact": artifact,
                "validation_mean_relative_error": report["validation_mean_relative_error"],
                "validation_mean_cosine": report["validation_mean_cosine"],
            }
        )
    summary = {
        "completed": len(completed),
        "skipped_existing": skipped,
        "seconds": time.perf_counter() - started,
        "results": completed,
    }
    (root / "reports" / "activation_sweep.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
