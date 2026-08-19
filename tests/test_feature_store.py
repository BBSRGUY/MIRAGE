import torch

from mirage.datasets import FeatureStore


def test_manifest_resume_without_duplicate(tmp_path):
    first = FeatureStore(tmp_path)
    assert first.append(
        "sample/0",
        {"value": torch.arange(4)},
        kind="activation",
        sample_id="sample",
        split="train",
        metadata={"step": 0},
    )
    first.mark_sample_complete("sample")
    resumed = FeatureStore(tmp_path)
    assert not resumed.append(
        "sample/0", {"value": torch.zeros(4)}, kind="activation", sample_id="sample", split="train"
    )
    assert resumed.sample_complete("sample")
    assert torch.equal(resumed.load("sample/0")["value"], torch.arange(4))
