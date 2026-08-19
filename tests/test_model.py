import torch

from mirage.config import MirageConfig
from mirage.model import MirageGenerator


def tiny_config() -> MirageConfig:
    return MirageConfig(
        frames=2,
        height=16,
        width=16,
        patch_size=8,
        hidden_size=32,
        depth=2,
        heads=4,
        basis_count=2,
        residual_rank=2,
        steps=2,
        text_tokens=8,
        vocabulary_size=128,
    )


def test_end_to_end_cpu_and_determinism():
    torch.manual_seed(1)
    model = MirageGenerator(tiny_config())
    first = model.generate("red kite", seed=2, device="cpu", use_cache=False)
    second = model.generate("red kite", seed=2, device="cpu", use_cache=False)
    assert first.video.shape == (1, 2, 3, 16, 16)
    assert first.audio.shape == (1, 640)
    assert torch.equal(first.video, second.video)
    assert first.telemetry.estimated_flops > 0


def test_shared_bank_is_actually_shared():
    model = MirageGenerator(tiny_config())
    assert model.blocks[0].attn.q.bank is model.blocks[1].attn.q.bank
    assert model.blocks[0].ff1.bank is model.bank


def test_predictive_cache_can_skip_blocks():
    config = tiny_config()
    config = MirageConfig(**{**config.to_dict(), "cache_threshold": 1e9})
    output = MirageGenerator(config).generate("still scene", seed=3, device="cpu", use_cache=True)
    assert output.telemetry.cache_hits > 0
