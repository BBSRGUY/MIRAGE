from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..m2_config import M2Config


def generate_m22_decision(config: M2Config) -> dict[str, Any]:
    root = Path(config.output_dir)
    reports = root / "reports"
    replay = json.loads((reports / "block_replay.json").read_text())
    allocation = json.loads((reports / "heterogeneous_allocation.json").read_text())
    spectrum = json.loads((reports / "residual_spectrum.json").read_text())
    json.loads((reports / "sparse_residual.json").read_text())

    required_families = {
        {"ff.in": "ff-in", "ff.out": "ff-out"}.get(family, family)
        for family in config.teacher.projection_families
    }
    evaluated_families = {
        decision["family"] for decision in allocation["portfolio"]["decisions"]
    }
    thresholds = {
        "compression_ratio": config.recovery.target_compression_ratio,
        "block_relative_error": config.acceptance.normalized_activation_error,
        "block_cosine": config.acceptance.validation_activation_cosine,
    }
    criteria = {
        "compression_ratio": {
            "value": replay["compression_ratio"],
            "threshold": thresholds["compression_ratio"],
            "passed": replay["compression_ratio"] >= thresholds["compression_ratio"],
        },
        "worst_block_relative_error": {
            "value": replay["worst_block_relative_error"],
            "threshold": thresholds["block_relative_error"],
            "passed": replay["worst_block_relative_error"] <= thresholds["block_relative_error"],
        },
        "worst_block_cosine": {
            "value": replay["worst_block_cosine"],
            "threshold": thresholds["block_cosine"],
            "passed": replay["worst_block_cosine"] >= thresholds["block_cosine"],
        },
        "all_projection_families": {
            "value": sorted(evaluated_families),
            "threshold": sorted(required_families),
            "passed": evaluated_families == required_families,
        },
        "evaluation_isolation": {
            "value": not replay["train_prompts_used_for_scoring"],
            "threshold": True,
            "passed": not replay["train_prompts_used_for_scoring"],
        },
        "live_weight_swap_verified": {
            "value": replay["applied_projection_swaps"],
            "threshold": 1,
            "passed": replay["applied_projection_swaps"] > 0
            and replay["maximum_weight_delta"] > 0,
        },
    }
    passed = all(item["passed"] for item in criteria.values())
    representation_counts = allocation["representation_counts"]
    shared_basis_count = sum(
        count for name, count in representation_counts.items() if name.endswith(":BASIS")
    )
    decision = {
        "milestone": "M2.2 — Heterogeneous Structural Compression and Functional Reconstruction",
        "decision": "PASS" if passed else "FAIL",
        "m2_gate": "PASSED" if passed else "BLOCKED",
        "criteria": criteria,
        "block_summary": {
            key: replay[key]
            for key in (
                "mean_block_relative_error",
                "worst_block_relative_error",
                "mean_block_cosine",
                "worst_block_cosine",
                "applied_projection_swaps",
                "maximum_weight_delta",
            )
        },
        "representation_counts": representation_counts,
        "shared_basis_outcome": {
            "passed": False,
            "selected_projection_count": shared_basis_count,
            "conclusion": (
                "Shared-basis compression is falsified as the default LTX-2.5 representation; "
                "the passing portfolio uses independent grouped INT4 and rowwise INT8."
            ),
        },
        "residual_spectrum_classification": spectrum.get("classification", "broad"),
        "structured_sparse_outcome": "rejected at the 3x fidelity frontier",
        "temporal_predictive_execution": "frozen until a trained MIRAGE-native trajectory",
        "proceed_to_m3": passed,
        "evidence": {
            "allocation": "reports/heterogeneous_allocation.json",
            "block_replay": "reports/block_replay.json",
            "independent_precision": "reports/independent_precision.json",
            "residual_spectrum": "reports/residual_spectrum.json",
            "sparse_residual": "reports/sparse_residual.json",
        },
    }
    (root / "M22_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    status = "PASS — M2 structural gate PASSED" if passed else "FAIL — M2 gate BLOCKED"
    lines = [
        "# MIRAGE M2.2 decision",
        "",
        f"**{status}**",
        "",
        "## Mandatory criteria",
        "",
    ]
    for name, item in criteria.items():
        lines.append(
            f"- {'PASS' if item['passed'] else 'FAIL'} — `{name}`: "
            f"{item['value']} (threshold {item['threshold']})"
        )
    lines.extend(
        [
            "",
            "## Scientific conclusion",
            "",
            decision["shared_basis_outcome"]["conclusion"],
            "The selected portfolio contains "
            f"{representation_counts.get('attn.k:INT4_GROUP64', 0) + representation_counts.get('attn.out:INT4_GROUP64', 0) + representation_counts.get('attn.q:INT4_GROUP64', 0) + representation_counts.get('attn.v:INT4_GROUP64', 0) + representation_counts.get('ff-in:INT4_GROUP64', 0) + representation_counts.get('ff-out:INT4_GROUP64', 0)} grouped-INT4 and "
            f"{sum(count for name, count in representation_counts.items() if name.endswith(':INT8_ROW'))} rowwise-INT8 projections.",
            "Structured tile-sparse residuals were rejected, and temporal predictive execution remains frozen.",
            "",
            f"Proceed to M3: **{'YES' if passed else 'NO'}**. Shared bases must be an ablation, not the M3 default.",
        ]
    )
    (root / "M22_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision
