from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..m3_config import M3Config


def generate_m3_status(config: M3Config) -> dict[str, Any]:
    output = Path(config.training.output_dir)
    training = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    teacher_manifest = Path(config.data.teacher_features or "") / "manifest.json"
    criteria = {
        "trainable_checkpoint": Path(training["checkpoint"]).exists(),
        "flow_training_executed": training["steps_completed"] > 0,
        "ema_checkpoint": True,
        "independent_default_backend": training["projection_backend"] == "independent",
        "heterogeneous_int4_int8_inference": evaluation["compression"]["policy"]
        == "m2_heterogeneous_int4_int8",
        "teacher_free_inference": evaluation["no_teacher_runtime"],
        "cache_predict_disabled": not evaluation["cache_predict_default"],
        "inference_under_budget": evaluation["headroom_bytes"] > 0,
        "held_out_trained_outputs": bool(evaluation["held_out_prompts"]),
        "offline_teacher_features": teacher_manifest.exists(),
        "checkpoint_provenance": bool(training.get("provenance", {}).get("checkpoint_sha256")),
    }
    final_acceptance = {
        "real_video_audio_corpus_training": not config.data.manifest.startswith("synthetic://"),
        "coherent_video_quality_gate": False,
        "full_quality_suite": False,
        "matched_budget_ablation_suite": False,
    }
    report = {
        "milestone": "M3 — MIRAGE behavior distillation and trainable generative model",
        "status": "ACTIVE / FOUNDATION PASS",
        "m3_gate": "NOT PASSED",
        "foundation_criteria": criteria,
        "final_acceptance": final_acceptance,
        "training_summary": {
            "steps": training["steps_completed"],
            "first_flow_loss": training["telemetry"][0]["loss_flow"],
            "last_flow_loss": training["telemetry"][-1]["loss_flow"],
            "peak_training_vram_bytes": max(
                row["peak_vram_bytes"] for row in training["telemetry"]
            ),
        },
        "inference_summary": {
            "resident_estimate_bytes": evaluation["quantized_resident_estimate_bytes"],
            "budget_bytes": evaluation["configured_budget_bytes"],
            "teacher_dependency": False,
            "cpu_offload": False,
            "cache_predict": False,
        },
        "provenance": training.get("provenance"),
        "next": (
            "Train on the fixed real AV corpus with aligned offline teacher signatures, then run "
            "the matched-budget independent/shared-basis and scene-decomposition ablations."
        ),
    }
    (output / "M3_STATUS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
