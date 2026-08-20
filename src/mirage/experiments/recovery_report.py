from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..m2_config import M2Config
from ..teacher.extraction import git_commit


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required M2.1 artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def generate_m21_report(config: M2Config) -> dict[str, Any]:
    root = Path(config.output_dir)
    metadata = _read(root / "metadata.json")
    compression = _read(root / "reports" / "adaptive_compression.json")
    activations = _read(root / "reports" / "adaptive_activation_sweep.json")
    delta = _read(root / "temporal" / "delta_spectrum.json")
    by_family = {row["family"]: row for row in activations["rows"]}
    required_families = {
        {"ff.in": "ff-in", "ff.out": "ff-out"}.get(family, family)
        for family in config.teacher.projection_families
    }
    evaluated_families = set(by_family)
    missing = sorted(required_families - evaluated_families)
    fit_rows = {row["family"]: row for row in compression["rows"]}
    total_original = total_compressed = 0
    for family in required_families & set(fit_rows):
        artifact = root / fit_rows[family]["artifact"]
        metrics = _read(artifact.with_suffix(".json"))
        total_original += int(metrics["original_params"])
        total_compressed += int(metrics["compressed_params"])
    compression_ratio = total_original / total_compressed if total_compressed else 0.0
    worst_error = max(
        (row["validation_mean_relative_error"] for row in by_family.values()), default=float("inf")
    )
    worst_cosine = min(
        (row["validation_mean_cosine"] for row in by_family.values()), default=float("-inf")
    )
    ff_evaluated = {"ff-in", "ff-out"}.issubset(evaluated_families)
    held_out = int(metadata["sample_counts"]["eval"])
    structural_criteria = {
        "activation_cosine": {
            "value": worst_cosine,
            "threshold": config.acceptance.validation_activation_cosine,
            "passed": worst_cosine >= config.acceptance.validation_activation_cosine,
        },
        "activation_error": {
            "value": worst_error,
            "threshold": config.acceptance.normalized_activation_error,
            "passed": worst_error <= config.acceptance.normalized_activation_error,
        },
        "compression_ratio": {
            "value": compression_ratio,
            "threshold": config.acceptance.compression_ratio,
            "passed": compression_ratio >= config.acceptance.compression_ratio,
        },
        "ff_families_evaluated": {"value": ff_evaluated, "threshold": True, "passed": ff_evaluated},
        "held_out_prompts": {"value": held_out, "threshold": 1, "passed": held_out >= 1},
    }
    structural_pass = all(item["passed"] for item in structural_criteria.values())
    delta_coverage = float(delta["decision"]["held_out_useful_coverage"])
    temporal_signal = delta_coverage >= config.temporal.delta_useful_coverage
    report = {
        "milestone": "M2.1 — Adaptive Structural Compression Recovery",
        "decision": "PASS" if structural_pass else "FAIL",
        "m2_gate": "BLOCKED",
        "structural_criteria": structural_criteria,
        "family_results": [by_family[name] for name in sorted(by_family)],
        "missing_families": missing,
        "temporal_recovery": {
            "phase": "M2.2",
            "oracle_delta_coverage": delta_coverage,
            "oracle_signal_passed": temporal_signal,
            "causal_delta_adapter_evaluated": False,
            "flop_reduction_measured": False,
            "gate_passed": False,
        },
        "proceed_to_m3": False,
        "provenance": {
            "commit_sha": git_commit(),
            "teacher": metadata["teacher"],
            "hardware": metadata["hardware"],
            "seed": metadata["seed"],
            "sample_counts": metadata["sample_counts"],
            "config": config.to_dict(),
        },
        "scope_warning": (
            "M2.1 is a held-out local activation study. M3 remains blocked until structural "
            "criteria pass and M2.2 demonstrates causal temporal execution at accepted error."
        ),
    }
    (root / "M21_DECISION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# MIRAGE M2.1 decision",
        "",
        f"**{report['decision']} — M2 gate BLOCKED**",
        "",
        "## Structural criteria",
        "",
    ]
    for name, item in structural_criteria.items():
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- {mark} — `{name}`: {item['value']} (threshold {item['threshold']})")
    lines.extend(
        [
            "",
            "## Temporal recovery",
            "",
            f"Oracle rank-delta coverage: {delta_coverage:.6g}.",
            "A causal delta adapter and measured FLOP reduction have not yet been evaluated.",
            "",
            "Proceed to M3: **NO**.",
        ]
    )
    (root / "M21_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
