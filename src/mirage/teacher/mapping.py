from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ProjectionSpec:
    """Stable MIRAGE name paired with an LTX-2.5 checkpoint/module path."""

    name: str
    checkpoint_key: str
    block_index: int
    family: str


LTX25_PROJECTION_PATHS: dict[str, str] = {
    "attn.q": "attn1.to_q",
    "attn.k": "attn1.to_k",
    "attn.v": "attn1.to_v",
    "attn.out": "attn1.to_out.0",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.out": "attn2.to_out.0",
    "ff.in": "ff.net.0.proj",
    "ff.out": "ff.net.2",
    "audio.attn.q": "audio_attn1.to_q",
    "audio.attn.k": "audio_attn1.to_k",
    "audio.attn.v": "audio_attn1.to_v",
    "audio.attn.out": "audio_attn1.to_out.0",
    "audio.cross_attn.q": "audio_attn2.to_q",
    "audio.cross_attn.k": "audio_attn2.to_k",
    "audio.cross_attn.v": "audio_attn2.to_v",
    "audio.cross_attn.out": "audio_attn2.to_out.0",
    "audio.ff.in": "audio_ff.net.0.proj",
    "audio.ff.out": "audio_ff.net.2",
    "av.audio_to_video.q": "audio_to_video_attn.to_q",
    "av.audio_to_video.k": "audio_to_video_attn.to_k",
    "av.audio_to_video.v": "audio_to_video_attn.to_v",
    "av.audio_to_video.out": "audio_to_video_attn.to_out.0",
    "av.video_to_audio.q": "video_to_audio_attn.to_q",
    "av.video_to_audio.k": "video_to_audio_attn.to_k",
    "av.video_to_audio.v": "video_to_audio_attn.to_v",
    "av.video_to_audio.out": "video_to_audio_attn.to_out.0",
}

_BLOCK_KEY = re.compile(
    r"^(?:model\.diffusion_model\.)?transformer_blocks\.(\d+)\.(.+)\.weight$"
)
_PATH_TO_FAMILY = {path: family for family, path in LTX25_PROJECTION_PATHS.items()}


def projection_specs_from_keys(keys: Iterable[str]) -> list[ProjectionSpec]:
    """Discover dense LTX-2.5 projection weights from a safetensors header."""

    specs: list[ProjectionSpec] = []
    for key in keys:
        match = _BLOCK_KEY.match(key)
        if match is None:
            continue
        block_index = int(match.group(1))
        module_path = match.group(2)
        family = _PATH_TO_FAMILY.get(module_path)
        if family is None:
            continue
        specs.append(
            ProjectionSpec(
                name=f"block.{block_index:02d}.{family}",
                checkpoint_key=key,
                block_index=block_index,
                family=family,
            )
        )
    return sorted(specs, key=lambda spec: (spec.block_index, spec.family))


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    current: nn.Module = root
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def map_ltx25_projections(blocks: list[nn.Module] | nn.ModuleList) -> Mapping[str, nn.Linear]:
    """Map official LTX-2.5 AV block modules into stable MIRAGE names."""

    mapped: dict[str, nn.Linear] = {}
    for index, block in enumerate(blocks):
        for family, path in LTX25_PROJECTION_PATHS.items():
            try:
                module = resolve_module(block, path)
            except (AttributeError, IndexError, TypeError):
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"expected Linear at LTX-2.5 block {index} {path}; "
                    f"found {type(module).__name__}"
                )
            mapped[f"block.{index:02d}.{family}"] = module
    return mapped


map_ltx_projections = map_ltx25_projections
