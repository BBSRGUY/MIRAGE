from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_sensitivity_map(
    candidates: list[dict[str, Any]],
    *,
    max_activation_error: float,
    min_cosine: float,
    min_compression_ratio: float,
) -> dict[str, Any]:
    """Rank layer projections by held-out local activation sensitivity."""
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        names = candidate["layer_names"]
        validation = candidate["validation"]
        for index, name in enumerate(names):
            row = validation[index]
            by_layer[name].append(
                {
                    "basis_count": candidate["basis_count"],
                    "rank": candidate["rank"],
                    "compression_ratio": candidate["compression_ratio"],
                    "activation_error": row["relative_activation_error"],
                    "activation_cosine": row["cosine_similarity"],
                    "residual_energy_ratio": candidate.get("residual_energy_ratio", 0.0),
                }
            )
    result: dict[str, Any] = {}
    for name, rows in by_layer.items():
        block = int(name.split(".")[1])
        maximum_block = max(int(key.split(".")[1]) for key in by_layer)
        depth = block / max(maximum_block, 1)
        best_fidelity = min(rows, key=lambda row: row["activation_error"])
        valid = [
            row
            for row in rows
            if row["activation_error"] <= max_activation_error
            and row["activation_cosine"] >= min_cosine
        ]
        recommended = (
            max(valid, key=lambda row: row["compression_ratio"]) if valid else best_fidelity
        )
        sensitivity = (
            best_fidelity["activation_error"]
            * (1 + depth)
            * (1 + best_fidelity["residual_energy_ratio"])
        )
        aggressive = bool(valid and recommended["compression_ratio"] >= min_compression_ratio)
        result[name] = {
            "sensitivity_score": sensitivity,
            "recommended_min_basis": recommended["basis_count"],
            "recommended_min_rank": recommended["rank"],
            "recommended_compression_ratio": recommended["compression_ratio"],
            "safe_for_aggressive_compression": aggressive,
            "best_activation_error": best_fidelity["activation_error"],
            "best_activation_cosine": best_fidelity["activation_cosine"],
        }
    return dict(sorted(result.items(), key=lambda item: item[1]["sensitivity_score"], reverse=True))
