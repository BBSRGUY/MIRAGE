import json

import torch

from mirage.config import MirageConfig
from mirage.datasets import StreamingAVDataset
from mirage.m3_config import M3Config, M3DataConfig, M3TrainingConfig
from mirage.model import MirageGenerator
from mirage.quantization import QuantizedLinear


def test_synthetic_stream_has_fixed_disjoint_splits():
    config = M3Config(
        model=MirageConfig(frames=2, height=16, width=16, patch_size=8),
        data=M3DataConfig(synthetic_samples=10, seed=11),
    )
    train = list(StreamingAVDataset(config, "train"))
    evaluation = list(StreamingAVDataset(config, "eval"))
    assert len(train) == 8
    assert len(evaluation) == 2
    assert {row["sample_id"] for row in train}.isdisjoint(
        {row["sample_id"] for row in evaluation}
    )


def test_m3_velocity_is_differentiable_with_independent_backend():
    config = MirageConfig(
        frames=2,
        height=16,
        width=16,
        patch_size=8,
        hidden_size=64,
        depth=2,
        heads=4,
        projection_backend="independent",
    )
    model = MirageGenerator(config)
    latent = torch.randn(1, config.video_tokens, config.hidden_size)
    velocity, states = model.predict_velocity(latent, ["moving sphere"], torch.tensor([0.5]))
    velocity.square().mean().backward()
    assert velocity.shape == latent.shape
    assert states["scene"].shape[1] == config.latent_height * config.latent_width
    assert model.blocks[0].attn.q.weight.grad is not None


def test_quantized_linear_uses_packed_int4_and_retains_output():
    torch.manual_seed(9)
    source = torch.nn.Linear(64, 32)
    value = torch.randn(2, 3, 64)
    quantized = QuantizedLinear(source, "INT4_GROUP64")
    relative = (quantized(value) - source(value)).norm() / source(value).norm()
    assert quantized.quantized.numel() == source.weight.numel() // 2
    assert relative < 0.2


def test_m3_config_round_trip(tmp_path):
    config = M3Config(
        model=MirageConfig(
            projection_backend="independent", cache_threshold=0.0, vram_budget_gb=20.0
        ),
        training=M3TrainingConfig(output_dir=str(tmp_path)),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    loaded = M3Config.from_json(path)
    loaded.validate()
    assert loaded.model.projection_backend == "independent"
