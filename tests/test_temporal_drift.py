import torch

from mirage.temporal.cache_analysis import cache_threshold_sweep, execution_policy, reuse_error
from mirage.temporal.drift import drift_metrics


def test_temporal_drift_and_reuse_policy():
    previous = torch.ones(2, 3, 4)
    current = previous * 1.01
    metrics = drift_metrics(previous, current)
    assert 0 < metrics.normalized_drift < 0.02
    replay = reuse_error(previous, current, torch.zeros_like(previous))
    sweep = cache_threshold_sweep([replay], [0.001, 0.1])
    assert sweep[0]["cache_hit_rate"] <= sweep[1]["cache_hit_rate"]
    assert execution_policy(0.01, 0.03, 0.02, 0.05) == "reuse"
    assert execution_policy(0.03, 0.04, 0.02, 0.05) == "predict"
    assert execution_policy(0.1, 0.1, 0.02, 0.05) == "execute"
