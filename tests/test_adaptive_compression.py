from mirage.compression.adaptive import allocate_residual_ranks, contiguous_behavior_groups


def test_behavior_groups_are_contiguous_and_complete():
    names = [f"block.{index:02d}.attn.q" for index in range(12)]
    sensitivity = {
        name: {
            "best_activation_error": index // 4,
            "best_activation_cosine": 1.0 - index / 100,
            "sensitivity_score": index + 1,
        }
        for index, name in enumerate(names)
    }
    groups = contiguous_behavior_groups(names, sensitivity, 3, minimum_group_size=2)
    assert [index for group in groups for index in group] == list(range(12))
    assert all(group == list(range(group[0], group[-1] + 1)) for group in groups)


def test_rank_allocation_respects_global_parameter_budget():
    names = [f"block.{index:02d}.attn.q" for index in range(8)]
    sensitivity = {name: {"sensitivity_score": index + 1} for index, name in enumerate(names)}
    ranks, report = allocate_residual_ranks(
        names,
        sensitivity,
        [2, 4, 8, 16],
        original_params=8000,
        shared_params=1000,
        rank_cost=100,
        target_compression_ratio=3.0,
    )
    assert 1000 + sum(ranks) * 100 <= 8000 / 3
    assert report["allocated_compression_ratio"] >= 3.0
    assert ranks[-1] >= ranks[0]
