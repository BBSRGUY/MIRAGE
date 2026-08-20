from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

from ..m2_config import M2Config


def _load_schemes(root: Path) -> dict[str, list[dict[str, Any]]]:
    schemes: dict[str, list[dict[str, Any]]] = {}
    for metadata_path in sorted((root / "reports" / "adaptive_fits").glob("*.json")):
        if "activation_fit" in metadata_path.stem:
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format") != "hierarchical_shared_basis_v1":
            continue
        activation_path = root / "reports" / f"activation_{metadata_path.stem}.json"
        if not activation_path.exists():
            continue
        activation = json.loads(activation_path.read_text())
        family = str(metadata["family"])
        shape = tuple(map(int, metadata["shape"]))
        fixed_params = (
            metadata["global_basis_count"]
            + len(metadata["groups"]) * metadata["group_basis_count"]
        ) * math.prod(shape)
        coefficient_params = metadata["global_basis_count"] + metadata["group_basis_count"]
        layers = []
        for index, name in enumerate(metadata["layer_names"]):
            rank = int(metadata["rank_per_layer"][index])
            layers.append(
                {
                    "layer": name,
                    "variable_bytes": 2
                    * (coefficient_params + rank * (shape[0] + shape[1])),
                    "activation_error": activation["validation"][index][
                        "relative_activation_error"
                    ],
                    "activation_cosine": activation["validation"][index][
                        "cosine_similarity"
                    ],
                }
            )
        schemes.setdefault(family, []).append(
            {
                "name": metadata_path.stem,
                "fixed_bytes": fixed_params * 2,
                "shape": shape,
                "layers": layers,
            }
        )
    return schemes


def run_heterogeneous_allocation(config: M2Config) -> dict[str, Any]:
    root = Path(config.output_dir)
    independent = json.loads(
        (root / "reports" / "independent_precision.json").read_text()
    )
    schemes = _load_schemes(root)
    families = sorted(schemes)
    if set(families) != set(independent["families"]):
        raise ValueError("basis and independent studies cover different families")
    int8 = {
        family: {
            layer["layer"]: layer
            for layer in independent["families"][family]["layers"]["INT8_ROW"]
        }
        for family in families
    }
    int4 = {
        family: {
            layer["layer"]: layer
            for layer in independent["families"][family]["layers"]["INT4_GROUP64"]
        }
        for family in families
    }
    # A basis bank is optional.  This pseudo-scheme lets the global solver reject
    # shared structure for a family and start from independent grouped INT4.
    for family in families:
        reference = schemes[family][0]
        schemes[family].append(
            {
                "name": "independent_int4_group64",
                "fixed_bytes": 0,
                "shape": reference["shape"],
                "layers": [
                    {
                        "layer": layer,
                        "variable_bytes": values["stored_bytes"],
                        "activation_error": values["activation_error"],
                        "activation_cosine": values["activation_cosine"],
                    }
                    for layer, values in sorted(int4[family].items())
                ],
            }
        )
    original_bytes = sum(
        len(schemes[family][0]["layers"]) * math.prod(schemes[family][0]["shape"]) * 2
        for family in families
    )
    budget = math.floor(original_bytes / config.recovery.target_compression_ratio)
    best: dict[str, Any] | None = None
    for selected in itertools.product(*(schemes[family] for family in families)):
        cost = sum(scheme["fixed_bytes"] for scheme in selected)
        decisions = []
        for family, scheme in zip(families, selected):
            for basis in scheme["layers"]:
                independent_layer = int8[family][basis["layer"]]
                is_int4 = scheme["name"] == "independent_int4_group64"
                if independent_layer["stored_bytes"] <= basis["variable_bytes"]:
                    choice = "INT8_ROW"
                    layer_cost = independent_layer["stored_bytes"]
                    error = independent_layer["activation_error"]
                    cosine = independent_layer["activation_cosine"]
                else:
                    choice = "INT4_GROUP64" if is_int4 else "BASIS"
                    layer_cost = basis["variable_bytes"]
                    error = basis["activation_error"]
                    cosine = basis["activation_cosine"]
                cost += layer_cost
                decisions.append(
                    {
                        "family": family,
                        "layer": basis["layer"],
                        "scheme": scheme["name"],
                        "choice": choice,
                        "bytes": layer_cost,
                        "activation_error": error,
                        "activation_cosine": cosine,
                        "int8": independent_layer,
                    }
                )
        if cost > budget:
            continue
        upgrades = []
        for index, decision in enumerate(decisions):
            if decision["choice"] == "INT8_ROW":
                continue
            independent_layer = decision["int8"]
            extra = independent_layer["stored_bytes"] - decision["bytes"]
            benefit = decision["activation_error"] ** 2 - independent_layer[
                "activation_error"
            ] ** 2
            if extra > 0 and benefit > 0:
                upgrades.append((benefit / extra, benefit, extra, index))
        for _ratio, _benefit, extra, index in sorted(upgrades, reverse=True):
            if cost + extra > budget:
                continue
            decision = decisions[index]
            independent_layer = decision["int8"]
            cost += extra
            decision.update(
                {
                    "choice": "INT8_ROW",
                    "bytes": independent_layer["stored_bytes"],
                    "activation_error": independent_layer["activation_error"],
                    "activation_cosine": independent_layer["activation_cosine"],
                }
            )
        squared_error = sum(decision["activation_error"] ** 2 for decision in decisions)
        candidate = {
            "stored_bytes": cost,
            "compression_ratio": original_bytes / cost,
            "rms_projection_error": math.sqrt(squared_error / len(decisions)),
            "worst_projection_error": max(d["activation_error"] for d in decisions),
            "worst_projection_cosine": min(d["activation_cosine"] for d in decisions),
            "decisions": [
                {key: value for key, value in decision.items() if key != "int8"}
                for decision in decisions
            ],
        }
        if best is None or candidate["rms_projection_error"] < best["rms_projection_error"]:
            best = candidate
    if best is None:
        raise ValueError("no heterogeneous portfolio meets the compression budget")
    counts: dict[str, int] = {}
    for decision in best["decisions"]:
        key = f"{decision['family']}:{decision['choice']}"
        counts[key] = counts.get(key, 0) + 1
    report = {
        "teacher": config.teacher.model_id,
        "objective": "minimum held-out projection RMS error under whole-model byte budget",
        "solver": "family-scheme enumeration plus sensitivity-greedy layer upgrades",
        "original_bytes": original_bytes,
        "budget_bytes": budget,
        "target_compression_ratio": config.recovery.target_compression_ratio,
        "portfolio": best,
        "representation_counts": counts,
        "scope_warning": (
            "Projection error is an allocation proxy. M2.2 acceptance uses held-out complete-block fidelity."
        ),
    }
    (root / "reports" / "heterogeneous_allocation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
