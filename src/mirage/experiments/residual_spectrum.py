from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..compression.factorization import load_factorization
from ..datasets import FeatureStore
from ..m2_config import M2Config


def matrix_residual_spectrum(
    residual: torch.Tensor,
    ranks: list[int],
    *,
    power_iterations: int,
    seed: int,
) -> dict[str, Any]:
    maximum = min(max(ranks), min(residual.shape))
    with torch.random.fork_rng(devices=[residual.device] if residual.is_cuda else []):
        torch.manual_seed(seed)
        _u, singular, _v = torch.svd_lowrank(
            residual.float(), q=maximum, niter=power_iterations
        )
    singular = singular.sort(descending=True).values
    total = residual.float().square().sum().clamp_min(1e-12)
    cumulative = singular.square().cumsum(0) / total
    curves = {}
    for rank in ranks:
        effective = min(rank, len(singular))
        captured = min(1.0, cumulative[effective - 1].item())
        curves[str(rank)] = {
            "effective_rank": effective,
            "captured_residual_energy": captured,
            "remaining_relative_error": max(0.0, 1.0 - captured) ** 0.5,
        }
    return {"shape": list(residual.shape), "ranks": curves}


def run_residual_spectrum(
    config: M2Config, device: str | torch.device | None = None
) -> dict[str, Any]:
    root = Path(config.output_dir)
    store = FeatureStore(root)
    device = torch.device(device or config.teacher.device)
    adaptive = json.loads((root / "reports" / "adaptive_compression.json").read_text())
    families: dict[str, Any] = {}
    for family_index, row in enumerate(adaptive["rows"]):
        source_artifact = root / row["artifact"]
        fitted_artifact = source_artifact.with_name(
            f"{source_artifact.stem}_activation_fit.safetensors"
        )
        artifact = fitted_artifact if fitted_artifact.exists() else source_artifact
        fit = load_factorization(artifact, device=device)
        names = list(fit.metrics["layer_names"])
        layers = []
        for index, name in enumerate(names):
            source = store.load(f"weight/{name}", device=device)["weight"].float()
            residual = source - fit.reconstruct_layer(index)
            layers.append(
                {
                    "layer": name,
                    **matrix_residual_spectrum(
                        residual,
                        config.recovery.residual_spectrum_ranks,
                        power_iterations=config.recovery.residual_spectrum_power_iterations,
                        seed=config.data.seed + family_index * len(names) + index,
                    ),
                }
            )
            del source, residual
        summary = {}
        for rank in config.recovery.residual_spectrum_ranks:
            values = [layer["ranks"][str(rank)]["captured_residual_energy"] for layer in layers]
            summary[str(rank)] = {
                "mean_captured_residual_energy": sum(values) / len(values),
                "minimum_captured_residual_energy": min(values),
                "maximum_captured_residual_energy": max(values),
            }
        families[row["family"]] = {
            "artifact": str(artifact),
            "summary": summary,
            "layers": layers,
            "classification": (
                "broad"
                if summary[str(min(256, max(config.recovery.residual_spectrum_ranks)))][
                    "mean_captured_residual_energy"
                ]
                < 0.9
                else "low_rank"
            ),
        }
        del fit
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "teacher": config.teacher.model_id,
        "method": "randomized_svd_with_exact_total_residual_energy",
        "power_iterations": config.recovery.residual_spectrum_power_iterations,
        "families": families,
    }
    (root / "reports" / "residual_spectrum.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
