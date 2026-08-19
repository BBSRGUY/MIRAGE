from __future__ import annotations

import hashlib
import re

import torch
from torch import nn


class HashTextEncoder(nn.Module):
    """Deterministic, trainable tokenizer/encoder with no external model residency."""

    def __init__(self, vocabulary_size: int, width: int, max_tokens: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocabulary_size, width)
        self.position = nn.Parameter(torch.randn(max_tokens, width) * 0.01)
        self.norm = nn.LayerNorm(width)

    def tokenize(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        rows = []
        for prompt in prompts:
            words = re.findall(r"[\w']+|[^\w\s]", prompt.lower())[: self.max_tokens]
            ids = [
                int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "little")
                % self.vocabulary_size
                for w in words
            ]
            ids += [0] * (self.max_tokens - len(ids))
            rows.append(ids)
        return torch.tensor(rows, device=device, dtype=torch.long)

    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        ids = self.tokenize(prompts, device)
        return self.norm(self.embedding(ids) + self.position[None])
