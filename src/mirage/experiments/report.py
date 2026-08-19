from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ..compression.sensitivity import build_sensitivity_map
from ..m2_config import M2Config
from ..teacher.extraction import git_commit


def score_decision(criteria: dict[str, dict[str, Any]]) -> str:
    """Return PASS only for complete evidence, PARTIAL for mixed evidence, otherwise FAIL."""
    if not criteria:
        raise ValueError("decision scoring requires at least one criterion")
    passed = sum(bool(item["passed"]) for item in criteria.values())
    return "PASS" if passed == len(criteria) else "PARTIAL" if passed else "FAIL"


def _read(path: Path, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required M2 artifact is missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _activation_candidates(root: Path) -> list[dict[str, Any]]:
    candidates = []
    for path in sorted((root / "reports").glob("activation_*.json")):
        report = _read(path)
        artifact = Path(report["artifact"])
        factor_metadata = _read(artifact.with_suffix(".json"))
        options = factor_metadata["options"]
        candidates.append(
            {
                **report,
                "basis_count": options["basis_count"],
                "rank": options["rank"],
                "compression_ratio": factor_metadata["compression_ratio"],
                "residual_energy_ratio": factor_metadata["residual_energy_ratio"],
                "family": factor_metadata["family"],
            }
        )
    if not candidates:
        raise FileNotFoundError("no activation validation reports found; run m2-activation-fit")
    return candidates


def generate_m2_report(config: M2Config) -> dict[str, Any]:
    root = Path(config.output_dir)
    metadata = _read(root / "metadata.json")
    temporal = _read(root / "temporal" / "temporal_redundancy.json")
    cache = _read(root / "temporal" / "cache_analysis.json")
    predictor = _read(root / "temporal" / "predictor_fit.json")
    scene = _read(root / "temporal" / "scene_motion.json")
    candidates = _activation_candidates(root)
    acceptance = config.acceptance
    best = max(
        candidates,
        key=lambda item: (
            item["validation_mean_cosine"] >= acceptance.validation_activation_cosine
            and item["validation_mean_relative_error"] <= acceptance.normalized_activation_error,
            item["compression_ratio"],
        ),
    )
    sensitivity = build_sensitivity_map(
        candidates,
        max_activation_error=acceptance.normalized_activation_error,
        min_cosine=acceptance.validation_activation_cosine,
        min_compression_ratio=acceptance.compression_ratio,
    )
    (root / "reports" / "sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2), encoding="utf-8"
    )
    residual_drifts = []
    for transitions in temporal["layers"].values():
        for hooks in transitions.values():
            if "block_residual" in hooks:
                residual_drifts.append(hooks["block_residual"]["normalized_drift"])
    max_cache = max(cache["threshold_sweep"], key=lambda row: row["threshold"])
    coverage = predictor["summary"]["reuse_percentage"] + predictor["summary"]["predict_percentage"]
    scene_energy = scene["summary"]["mean_rank1_explained_energy"]
    criteria = {
        "validation_activation_cosine": {
            "value": best["validation_mean_cosine"],
            "threshold": acceptance.validation_activation_cosine,
            "passed": best["validation_mean_cosine"] >= acceptance.validation_activation_cosine,
        },
        "normalized_activation_error": {
            "value": best["validation_mean_relative_error"],
            "threshold": acceptance.normalized_activation_error,
            "passed": best["validation_mean_relative_error"]
            <= acceptance.normalized_activation_error,
        },
        "compression_ratio": {
            "value": best["compression_ratio"],
            "threshold": acceptance.compression_ratio,
            "passed": best["compression_ratio"] >= acceptance.compression_ratio,
        },
        "reuse_predict_coverage": {
            "value": coverage,
            "threshold": acceptance.reuse_predict_coverage,
            "passed": coverage >= acceptance.reuse_predict_coverage,
        },
        "scene_low_rank_energy": {
            "value": scene_energy,
            "threshold": acceptance.scene_low_rank_energy,
            "passed": scene_energy >= acceptance.scene_low_rank_energy,
        },
    }
    decision = score_decision(criteria)
    family_summary: dict[str, list[float]] = {}
    for candidate in candidates:
        family_summary.setdefault(candidate["family"], []).append(
            candidate["validation_mean_relative_error"]
        )
    report = {
        "milestone": "M2 — Teacher Structural Compressibility & Temporal Redundancy Study",
        "decision": decision,
        "criteria": criteria,
        "questions": {
            "shared_basis_promising": all(
                criteria[key]["passed"]
                for key in (
                    "validation_activation_cosine",
                    "normalized_activation_error",
                    "compression_ratio",
                )
            ),
            "best_acceptable_compression_ratio": best["compression_ratio"],
            "best_projection_families": sorted(
                family_summary, key=lambda family: statistics.mean(family_summary[family])
            ),
            "most_sensitive_layers": list(sensitivity)[:10],
            "median_adjacent_residual_drift": statistics.median(residual_drifts),
            "cache_rate_at_largest_threshold": max_cache["cache_hit_rate"],
            "reuse_predict_coverage": coverage,
            "scene_rank1_energy": scene_energy,
            "proceed_to_full_distillation": decision == "PASS",
        },
        "best_candidate": {
            key: best[key]
            for key in (
                "family",
                "basis_count",
                "rank",
                "compression_ratio",
                "validation_mean_relative_error",
                "validation_mean_cosine",
                "artifact",
            )
        },
        "scientific_scope": {
            "weight_reconstruction": True,
            "held_out_activation_reconstruction": True,
            "local_teacher_behavior": True,
            "full_generation_quality": False,
            "warning": "This decision does not establish perceptual equivalence.",
        },
        "provenance": {
            "commit_sha": git_commit(),
            "teacher": metadata["teacher"],
            "seed": metadata["seed"],
            "hardware": metadata["hardware"],
            "dtype": metadata["teacher"]["dtype"],
            "config": config.to_dict(),
            "sample_counts": metadata["sample_counts"],
        },
    }
    json_path = root / "M2_DECISION.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# MIRAGE M2 decision",
        "",
        f"**{decision}**",
        "",
        "## Acceptance criteria",
        "",
    ]
    for name, item in criteria.items():
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- {mark} — `{name}`: {item['value']:.6g} (threshold {item['threshold']})")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            (
                "This decision covers trained-teacher weight reconstruction, held-out local activation fidelity, "
                "and internal temporal behavior. It does **not** demonstrate full-video perceptual equivalence."
            ),
        ]
    )
    (root / "M2_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
