from __future__ import annotations

import io
import json
import math
import random
import tarfile
from glob import glob
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av
import torch
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.utils.data import IterableDataset, get_worker_info

from ..m3_config import M3Config


def _load_tensor(path: Path, key: str) -> torch.Tensor:
    if path.suffix == ".safetensors":
        values = load_file(str(path), device="cpu")
    else:
        values = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(values, torch.Tensor):
            return values
    if key not in values:
        raise KeyError(f"{path} does not contain {key!r}")
    return values[key]


class StreamingAVDataset(IterableDataset):
    """Deterministic JSONL streaming dataset with rank/worker sharding.

    Each row contains `sample_id`, `split`, `prompt`, `video`, and optional
    `audio`/`teacher_feature`. Tensor files may be PyTorch or Safetensors.
    """

    def __init__(self, config: M3Config, split: str):
        super().__init__()
        self.config = config
        self.split = split

    def _shard(self) -> tuple[int, int]:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        return rank * workers + worker_id, world * workers

    def _synthetic(self) -> Iterator[dict[str, Any]]:
        c = self.config.model
        shard, shards = self._shard()
        count = self.config.data.synthetic_samples
        split_at = max(1, int(count * 0.8))
        indices = range(0, split_at) if self.split == "train" else range(split_at, count)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, c.height), torch.linspace(-1, 1, c.width), indexing="ij"
        )
        for index in indices:
            if index % shards != shard:
                continue
            generator = torch.Generator().manual_seed(self.config.data.seed + index)
            phase = torch.rand((), generator=generator).item() * math.tau
            frames = []
            envelope = []
            for frame in range(c.frames):
                angle = phase + frame * math.tau / max(c.frames, 2)
                cx, cy = 0.55 * math.cos(angle), 0.55 * math.sin(angle)
                blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.12)
                frames.append(torch.stack((blob, blob * 0.6, 1 - blob * 0.7)).mul(2).sub(1))
                envelope.append(0.25 + 0.75 * blob.mean().item())
            samples = c.frames * 320
            time = torch.arange(samples) / 16_000
            audio = torch.sin(math.tau * (220 + index % 5 * 40) * time)
            audio *= torch.tensor(envelope).repeat_interleave(320)
            yield {
                "sample_id": f"synthetic-{index:06d}",
                "split": self.split,
                "prompt": "a colored sphere moves smoothly in a circular path",
                "video": torch.stack(frames),
                "audio": audio,
                "teacher_feature": None,
            }

    def _manifest_rows(self) -> Iterator[dict[str, Any]]:
        manifest = Path(self.config.data.manifest)
        root = manifest.parent
        shard, shards = self._shard()
        rng = random.Random(self.config.data.seed)

        def materialize(row: dict[str, Any]) -> dict[str, Any]:
            video_path = root / row["video"]
            audio_path = root / row["audio"] if row.get("audio") else None
            feature_path = root / row["teacher_feature"] if row.get("teacher_feature") else None
            return {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "prompt": row["prompt"],
                "video": _load_tensor(video_path, "video").float(),
                "audio": _load_tensor(audio_path, "audio").float() if audio_path else None,
                "teacher_feature": (
                    _load_tensor(feature_path, "signature").float() if feature_path else None
                ),
            }

        buffer: list[dict[str, Any]] = []
        selected_index = 0
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] != self.split:
                    continue
                belongs = selected_index % shards == shard
                selected_index += 1
                if not belongs:
                    continue
                if self.split != "train":
                    yield materialize(row)
                    continue
                buffer.append(row)
                if len(buffer) >= self.config.data.shuffle_buffer:
                    yield materialize(buffer.pop(rng.randrange(len(buffer))))
        while buffer:
            yield materialize(buffer.pop(rng.randrange(len(buffer))))

    def _decode_video(self, payload: bytes) -> torch.Tensor:
        frames = []
        with av.open(io.BytesIO(payload)) as container:
            for frame in container.decode(video=0):
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
        if not frames:
            raise ValueError("WebDataset sample contains no decodable video frames")
        indices = torch.linspace(0, len(frames) - 1, self.config.model.frames).round().long()
        selected = torch.stack([frames[index].float() / 127.5 - 1 for index in indices])
        return F.interpolate(
            selected,
            size=(self.config.model.height, self.config.model.width),
            mode="bilinear",
            align_corners=False,
        )

    def _decode_audio(self, payload: bytes) -> torch.Tensor | None:
        chunks = []
        try:
            with av.open(io.BytesIO(payload)) as container:
                if not container.streams.audio:
                    return None
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16_000)
                for frame in container.decode(audio=0):
                    converted = resampler.resample(frame)
                    converted = converted if isinstance(converted, list) else [converted]
                    for item in converted:
                        if item is not None:
                            chunks.append(torch.from_numpy(item.to_ndarray()).float().flatten())
        except (av.error.FFmpegError, EOFError):
            return None
        if not chunks:
            return None
        audio = torch.cat(chunks)
        required = self.config.model.frames * 320
        return F.interpolate(audio[None, None], size=required, mode="linear", align_corners=False)[
            0, 0
        ]

    def _webdataset_rows(self) -> Iterator[dict[str, Any]]:
        pattern = self.config.data.manifest.removeprefix("webdataset://")
        split = "val" if self.split == "eval" else self.split
        pattern = pattern.replace("{split}", split)
        paths = [Path(path) for path in sorted(glob(pattern))]
        if not paths:
            raise FileNotFoundError(f"no WebDataset shards match {pattern}")
        rng = random.Random(self.config.data.seed)
        if self.split == "train":
            rng.shuffle(paths)
        shard, shards = self._shard()
        for path_index, path in enumerate(paths):
            if path_index % shards != shard:
                continue
            with tarfile.open(path, "r|*") as archive:
                current_id = None
                parts: dict[str, bytes] = {}
                for member in archive:
                    if not member.isfile():
                        continue
                    sample_id = Path(member.name).stem
                    if current_id is not None and sample_id != current_id:
                        yield self._materialize_webdataset(current_id, parts, split)
                        parts = {}
                    current_id = sample_id
                    handle = archive.extractfile(member)
                    if handle is not None:
                        parts[Path(member.name).suffix] = handle.read()
                if current_id is not None:
                    yield self._materialize_webdataset(current_id, parts, split)

    def _materialize_webdataset(
        self, sample_id: str, parts: dict[str, bytes], split: str
    ) -> dict[str, Any]:
        if ".mp4" not in parts or ".txt" not in parts or ".json" not in parts:
            raise ValueError(f"incomplete WebDataset sample: {sample_id}")
        metadata = json.loads(parts[".json"])
        return {
            "sample_id": sample_id,
            "split": split,
            "prompt": parts[".txt"].decode("utf-8").strip(),
            "video": self._decode_video(parts[".mp4"]),
            "audio": self._decode_audio(parts[".mp4"]),
            "teacher_feature": None,
            "metadata": metadata,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.config.data.manifest.startswith("synthetic://"):
            yield from self._synthetic()
        elif self.config.data.manifest.startswith("webdataset://"):
            yield from self._webdataset_rows()
        else:
            yield from self._manifest_rows()


def collate_av(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audio_enabled = all(row["audio"] is not None for row in rows)
    features_enabled = all(row["teacher_feature"] is not None for row in rows)
    return {
        "sample_id": [row["sample_id"] for row in rows],
        "prompt": [row["prompt"] for row in rows],
        "video": torch.stack([row["video"] for row in rows]),
        "audio": torch.stack([row["audio"] for row in rows]) if audio_enabled else None,
        "teacher_feature": (
            torch.stack([row["teacher_feature"] for row in rows]) if features_enabled else None
        ),
    }
