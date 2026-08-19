from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptRecord:
    sample_id: str
    text: str
    split: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def load_prompt_split(
    path: str | Path, train_count: int, eval_count: int, seed: int
) -> list[PromptRecord]:
    """Load, deterministically shuffle, and split a newline-delimited prompt set."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line and not line.startswith("#")]
    required = train_count + eval_count
    if len(prompts) < required:
        raise ValueError(f"prompt file contains {len(prompts)} prompts; {required} required")
    order = list(range(len(prompts)))
    random.Random(seed).shuffle(order)
    records = []
    for index, source_index in enumerate(order[:required]):
        split = "train" if index < train_count else "eval"
        records.append(PromptRecord(f"{split}-{index:05d}", prompts[source_index], split))
    return records
