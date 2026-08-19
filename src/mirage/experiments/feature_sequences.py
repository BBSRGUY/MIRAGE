from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from ..datasets import FeatureStore


@dataclass(frozen=True)
class BlockStepFeatures:
    sample_id: str
    split: str
    block_index: int
    step_index: int
    timestep: float
    block_input: torch.Tensor
    block_output: torch.Tensor
    residual: torch.Tensor
    attention: torch.Tensor | None


def load_block_sequences(
    store: FeatureStore, split: str | None = None
) -> dict[tuple[str, int], list[BlockStepFeatures]]:
    """Join independently streamed hook records into adjacent-step block sequences."""
    joined: dict[tuple[str, int, int], dict[str, object]] = defaultdict(dict)
    for record in store.records(kind="activation", split=split):
        hook = record.metadata.get("hook")
        if hook not in {"block_input", "block_output", "block_residual", "attention_output"}:
            continue
        # M2's temporal policy is evaluated on the video stream. Audio features are
        # retained in the store for separate AV analyses, but must not overwrite video.
        if record.metadata.get("modality", "video") != "video":
            continue
        block = int(record.metadata["block_index"])
        step = int(record.metadata["step_index"])
        key = (record.sample_id, block, step)
        joined[key][str(hook)] = store.load(record)["value"]
        joined[key]["split"] = record.split
        joined[key]["timestep"] = float(record.metadata["timestep"])
    sequences: dict[tuple[str, int], list[BlockStepFeatures]] = defaultdict(list)
    for (sample_id, block, step), values in joined.items():
        required = {"block_input", "block_output", "block_residual"}
        if not required.issubset(values):
            continue
        sequences[(sample_id, block)].append(
            BlockStepFeatures(
                sample_id=sample_id,
                split=str(values["split"]),
                block_index=block,
                step_index=step,
                timestep=float(values["timestep"]),
                block_input=values["block_input"],  # type: ignore[arg-type]
                block_output=values["block_output"],  # type: ignore[arg-type]
                residual=values["block_residual"],  # type: ignore[arg-type]
                attention=values.get("attention_output"),  # type: ignore[arg-type]
            )
        )
    for values in sequences.values():
        values.sort(key=lambda item: item.step_index)
    if not sequences:
        raise ValueError("feature store contains no complete block feature sequences")
    return dict(sequences)
