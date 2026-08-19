from __future__ import annotations

from collections import defaultdict

import torch

from ..datasets import FeatureStore


def projection_family(normalized_name: str) -> str:
    parts = normalized_name.split(".")
    if len(parts) < 4 or parts[0] != "block" or not parts[1].isdigit():
        raise ValueError(f"invalid normalized projection name: {normalized_name}")
    family = ".".join(parts[2:])
    return {"ff.in": "ff-in", "ff.out": "ff-out"}.get(family, family)


def load_weight_families(
    store: FeatureStore, device: str | torch.device = "cpu"
) -> dict[str, dict[str, torch.Tensor]]:
    """Load extracted weights grouped by semantic family and exact shape."""
    grouped: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for record in store.records(kind="weight"):
        name = str(record.metadata["name"])
        family = projection_family(name)
        tensor = store.load(record, device=device)["weight"]
        grouped[family][name] = tensor
    if not grouped:
        raise ValueError("feature store contains no extracted teacher weights")
    return dict(grouped)


def compatible_groups(
    weights: dict[str, torch.Tensor],
) -> list[tuple[tuple[int, ...], list[tuple[str, torch.Tensor]]]]:
    shapes: dict[tuple[int, ...], list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for name, tensor in weights.items():
        shapes[tuple(tensor.shape)].append((name, tensor))
    return sorted(shapes.items(), key=lambda item: item[0])
