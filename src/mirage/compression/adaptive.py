from __future__ import annotations

import math
from typing import Any, Sequence

import torch


def contiguous_behavior_groups(
    layer_names: Sequence[str],
    sensitivity: dict[str, Any],
    group_count: int,
    *,
    minimum_group_size: int = 4,
) -> list[list[int]]:
    """Find contiguous depth groups minimizing within-group behavior-signature variance."""
    layers = len(layer_names)
    if group_count < 1 or group_count * minimum_group_size > layers:
        raise ValueError("invalid group_count or minimum_group_size")
    if not all(name in sensitivity for name in layer_names):
        boundaries = [round(index * layers / group_count) for index in range(group_count + 1)]
        return [list(range(boundaries[i], boundaries[i + 1])) for i in range(group_count)]
    features = torch.tensor(
        [
            [
                sensitivity[name]["best_activation_error"],
                1.0 - sensitivity[name]["best_activation_cosine"],
                sensitivity[name]["sensitivity_score"],
            ]
            for name in layer_names
        ],
        dtype=torch.float64,
    )
    features = (features - features.mean(0)) / features.std(0).clamp_min(1e-12)
    prefix = torch.cat((torch.zeros(1, features.shape[1]), features.cumsum(0)))
    prefix_square = torch.cat((torch.zeros(1), features.square().sum(1).cumsum(0)))

    def cost(start: int, stop: int) -> float:
        count = stop - start
        total = prefix[stop] - prefix[start]
        return float(prefix_square[stop] - prefix_square[start] - total.square().sum() / count)

    infinity = float("inf")
    dp = [[infinity] * (layers + 1) for _ in range(group_count + 1)]
    previous = [[-1] * (layers + 1) for _ in range(group_count + 1)]
    dp[0][0] = 0.0
    for groups in range(1, group_count + 1):
        lower = groups * minimum_group_size
        for stop in range(lower, layers + 1):
            start_min = (groups - 1) * minimum_group_size
            for start in range(start_min, stop - minimum_group_size + 1):
                candidate = dp[groups - 1][start] + cost(start, stop)
                if candidate < dp[groups][stop]:
                    dp[groups][stop] = candidate
                    previous[groups][stop] = start
    groups: list[list[int]] = []
    stop = layers
    for count in range(group_count, 0, -1):
        start = previous[count][stop]
        if start < 0:
            raise RuntimeError("failed to construct contiguous behavior groups")
        groups.append(list(range(start, stop)))
        stop = start
    return list(reversed(groups))


def allocate_residual_ranks(
    layer_names: Sequence[str],
    sensitivity: dict[str, Any],
    rank_tiers: Sequence[int],
    *,
    original_params: int,
    shared_params: int,
    rank_cost: int,
    target_compression_ratio: float,
) -> tuple[list[int], dict[str, Any]]:
    """Allocate rank by sensitivity, then trim least-sensitive layers to the byte budget."""
    if len(rank_tiers) != 4 or sorted(rank_tiers) != list(rank_tiers):
        raise ValueError("rank_tiers must contain four ascending values")
    scores = [float(sensitivity.get(name, {}).get("sensitivity_score", 1.0)) for name in layer_names]
    order = sorted(range(len(layer_names)), key=lambda index: scores[index])
    ranks = [0] * len(layer_names)
    for position, index in enumerate(order):
        quantile = min(3, (4 * position) // max(len(order), 1))
        ranks[index] = int(rank_tiers[quantile])
    maximum_params = math.floor(original_params / target_compression_ratio)
    choices = [0, *map(int, rank_tiers)]
    while shared_params + sum(ranks) * rank_cost > maximum_params:
        changed = False
        for index in order:
            current = choices.index(ranks[index])
            if current:
                ranks[index] = choices[current - 1]
                changed = True
                break
        if not changed:
            raise ValueError("shared bases alone exceed the target compression budget")
    compressed = shared_params + sum(ranks) * rank_cost
    return ranks, {
        "target_compression_ratio": target_compression_ratio,
        "allocated_compression_ratio": original_params / compressed,
        "allocated_params": compressed,
        "rank_histogram": {str(rank): ranks.count(rank) for rank in sorted(set(ranks))},
        "sensitivity_fallback_count": sum(name not in sensitivity for name in layer_names),
    }
