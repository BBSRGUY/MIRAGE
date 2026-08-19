import pytest
from torch import nn

from mirage.m2_config import TeacherConfig
from mirage.teacher import get_teacher_adapter, registered_teachers
from mirage.teacher.mapping import map_ltx_projections


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.to_k = nn.Linear(4, 4)
        self.to_v = nn.Linear(4, 4)
        self.to_out = nn.ModuleList([nn.Linear(4, 4), nn.Dropout()])


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        entry = nn.Module()
        entry.proj = nn.Linear(4, 8)
        self.net = nn.ModuleList([entry, nn.SiLU(), nn.Linear(8, 4)])


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = Attention()
        self.attn2 = Attention()
        self.ff = FeedForward()


def test_normalized_ltx_names():
    mapping = map_ltx_projections(nn.ModuleList([Block(), Block()]))
    assert "block.00.attn.q" in mapping
    assert "block.01.ff.out" in mapping
    assert all(isinstance(module, nn.Linear) for module in mapping.values())


def test_registry_is_explicit():
    assert "ltx25" in registered_teachers()
    assert get_teacher_adapter("ltx25", TeacherConfig()).model_identifier.startswith("Lightricks")
    with pytest.raises(ValueError, match="unsupported teacher"):
        get_teacher_adapter("unknown", TeacherConfig())
