from __future__ import annotations

import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..datasets import FeatureStore
from ..m2_config import M2Config
from ..temporal.drift import drift_metrics
from .feature_sequences import load_block_sequences


def run_temporal_probe(config: M2Config) -> dict[str, Any]:
    store = FeatureStore(config.output_dir)
    sequences = load_block_sequences(store)
    matrix: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for (_sample, block), steps in sequences.items():
        for previous, current in pairwise(steps):
            transition = f"{previous.step_index}->{current.step_index}"
            pairs = {
                "block_input": (previous.block_input, current.block_input),
                "block_residual": (previous.residual, current.residual),
                "block_output": (previous.block_output, current.block_output),
            }
            if previous.attention is not None and current.attention is not None:
                pairs["attention_output"] = (previous.attention, current.attention)
            for name, pair in pairs.items():
                matrix[str(block)][transition][name].append(drift_metrics(*pair).to_dict())
    aggregated: dict[str, Any] = {}
    for block, transitions in matrix.items():
        aggregated[block] = {}
        for transition, hooks in transitions.items():
            aggregated[block][transition] = {}
            for hook, rows in hooks.items():
                aggregated[block][transition][hook] = {
                    key: sum(row[key] for row in rows) / len(rows) for key in rows[0]
                }
    report = {
        "teacher": config.teacher.model_id,
        "metric_definition": "RMS(A-B) / max(RMS(A), epsilon)",
        "scope_warning": "Internal adjacent-step fidelity, not perceptual equivalence.",
        "layers": aggregated,
    }
    target = Path(config.output_dir) / "temporal" / "temporal_redundancy.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
