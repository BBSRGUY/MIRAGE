import pytest

from mirage.config import MirageConfig


def test_shape_contract():
    config = MirageConfig(frames=4, height=32, width=32, patch_size=8, hidden_size=64, heads=4)
    config.validate()
    assert config.video_tokens == 64


def test_invalid_head_width():
    with pytest.raises(ValueError):
        MirageConfig(hidden_size=65, heads=4).validate()
