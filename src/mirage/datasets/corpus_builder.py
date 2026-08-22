from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import av
import torch
from torch.nn import functional as F

from ..m3_data_config import M3CorpusConfig
from ..training.provenance import canonical_json_sha256, sha256_file


def probe_media(path: str | Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    raw = json.loads(result.stdout)
    video = next((stream for stream in raw["streams"] if stream["codec_type"] == "video"), None)
    audio = next((stream for stream in raw["streams"] if stream["codec_type"] == "audio"), None)
    if video is None:
        raise ValueError("media has no video stream")
    duration = float(raw.get("format", {}).get("duration") or video.get("duration") or 0)
    return {
        "duration": duration,
        "width": int(video["width"]),
        "height": int(video["height"]),
        "video_codec": video.get("codec_name"),
        "has_audio": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
    }


def motion_signature(
    path: str | Path, maximum_frames: int = 12, maximum_decode_frames: int = 192
) -> tuple[float, str]:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            value = torch.from_numpy(frame.to_ndarray(format="gray"))[None, None].float() / 255
            value = F.interpolate(value, size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
            frames.append(value)
            if len(frames) >= maximum_decode_frames:
                break
    if len(frames) < 2:
        return 0.0, ""
    indices = torch.linspace(0, len(frames) - 1, min(maximum_frames, len(frames))).round().long()
    stacked = torch.stack([frames[index] for index in indices])
    score = (stacked[1:] - stacked[:-1]).abs().mean().item()
    digest_indices = sorted({0, len(stacked) // 2, len(stacked) - 1})
    digest = hashlib.sha256()
    for index in digest_indices:
        digest.update((stacked[index] * 255).byte().numpy().tobytes())
    return score, digest.hexdigest()


def _load_selection(config: M3CorpusConfig) -> dict[str, dict[str, Any]]:
    path = Path(config.work_dir) / "selection" / "selected.jsonl"
    return {
        row["sample_id"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def _metadata_id(path: Path) -> str | None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for container in (raw, raw.get("additional_columns", {}), raw.get("meta", {})):
        if isinstance(container, dict) and container.get("mirage_id"):
            value = container["mirage_id"]
            if isinstance(value, list):
                value = value[0]
            return str(value)
    return None


def normalize_downloaded_corpus(config: M3CorpusConfig) -> dict[str, Any]:
    selected = _load_selection(config)
    source_root = Path(config.downloaded_dir)
    destination_root = Path(config.normalized_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    rejects: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    exact_hashes: set[str] = set()
    visual_hashes: set[str] = set()
    accepted_ids: set[str] = set()
    for metadata_path in sorted(destination_root.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sample_id = metadata.get("sample_id")
        content_hash = metadata.get("content_sha256")
        visual_hash = metadata.get("visual_sha256")
        video_path = destination_root / f"{sample_id}.mp4"
        text_path = destination_root / f"{sample_id}.txt"
        if not sample_id or not content_hash or not visual_hash:
            continue
        if not video_path.is_file() or not text_path.is_file():
            continue
        accepted.append(metadata)
        accepted_ids.add(sample_id)
        exact_hashes.add(content_hash)
        visual_hashes.add(visual_hash)
    resumed_samples = len(accepted)
    for video_path in sorted(source_root.rglob("*.mp4")):
        metadata_path = video_path.with_suffix(".json")
        text_path = video_path.with_suffix(".txt")
        sample_id = _metadata_id(metadata_path) if metadata_path.exists() else None
        if sample_id not in selected:
            rejects["selection_mismatch"] += 1
            continue
        if sample_id in accepted_ids:
            continue
        row = selected[sample_id]
        try:
            probe = probe_media(video_path)
        except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            rejects["probe_failure"] += 1
            continue
        if not config.duration_min_s - 0.25 <= probe["duration"] <= config.duration_max_s + 0.5:
            rejects["duration"] += 1
            continue
        if probe["width"] < config.min_download_width or probe["height"] < config.min_download_height:
            rejects["resolution"] += 1
            continue
        if config.require_audio and not probe["has_audio"]:
            rejects["audio_missing"] += 1
            continue
        motion, visual_hash = motion_signature(video_path)
        if not config.motion_difference_min <= motion <= config.motion_difference_max:
            rejects["motion"] += 1
            continue
        destination = destination_root / f"{sample_id}.mp4"
        command = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            (
                f"scale={config.normalized_size}:{config.normalized_size}:"
                "force_original_aspect_ratio=increase,"
                f"crop={config.normalized_size}:{config.normalized_size},fps={config.normalized_fps}"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "1",
            "-ar",
            str(config.audio_rate),
            "-b:a",
            "64k",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=180)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            rejects["normalization_failure"] += 1
            destination.unlink(missing_ok=True)
            continue
        content_hash = sha256_file(destination)
        if content_hash in exact_hashes or visual_hash in visual_hashes:
            rejects["duplicate"] += 1
            destination.unlink(missing_ok=True)
            continue
        exact_hashes.add(content_hash)
        visual_hashes.add(visual_hash)
        caption = (
            text_path.read_text(encoding="utf-8").strip()
            if text_path.exists()
            else row["caption"]
        )
        metadata = {
            **row,
            "caption": caption,
            "normalized": probe_media(destination),
            "motion_difference": motion,
            "visual_sha256": visual_hash,
            "content_sha256": content_hash,
            "source_probe": probe,
            "license": "Panda-70M research use; source-video licenses remain individually binding",
        }
        (destination_root / f"{sample_id}.txt").write_text(caption + "\n", encoding="utf-8")
        (destination_root / f"{sample_id}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        accepted.append(metadata)
        accepted_ids.add(sample_id)
    accepted.sort(key=lambda row: row["sample_id"])
    manifest = destination_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted), encoding="utf-8"
    )
    report = {
        "format": "mirage_m3_normalization_v1",
        "config_sha256": canonical_json_sha256(config.to_dict()),
        "selection_sha256": sha256_file(Path(config.work_dir) / "selection" / "selected.jsonl"),
        "resumed_samples": resumed_samples,
        "accepted": len(accepted),
        "reject_counts": dict(rejects),
        "split_counts": dict(Counter(row["split"] for row in accepted)),
        "motion_bucket_counts": dict(Counter(row["motion_bucket"] for row in accepted)),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    (destination_root / "normalization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _add_bytes(tar: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(value))


def shard_normalized_corpus(config: M3CorpusConfig) -> dict[str, Any]:
    source = Path(config.normalized_dir)
    destination = Path(config.shards_dir)
    destination.mkdir(parents=True, exist_ok=True)
    existing = sorted(destination.glob("*.tar"))
    if existing:
        raise FileExistsError(
            f"refusing to mix a new corpus with {len(existing)} existing shard(s) in {destination}"
        )
    rows = [
        json.loads(line)
        for line in (source / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    shards = []
    for split in ("train", "val", "test"):
        split_rows = sorted(
            (row for row in rows if row["split"] == split), key=lambda row: row["sample_id"]
        )
        for shard_index, start in enumerate(range(0, len(split_rows), config.shard_size)):
            shard_path = destination / f"{split}-{shard_index:06d}.tar"
            members = split_rows[start : start + config.shard_size]
            with tarfile.open(shard_path, "w") as tar:
                for row in members:
                    sample_id = row["sample_id"]
                    _add_bytes(tar, f"{sample_id}.mp4", (source / f"{sample_id}.mp4").read_bytes())
                    _add_bytes(tar, f"{sample_id}.txt", (source / f"{sample_id}.txt").read_bytes())
                    _add_bytes(tar, f"{sample_id}.json", json.dumps(row, sort_keys=True).encode())
            shards.append(
                {
                    "split": split,
                    "path": str(shard_path),
                    "samples": len(members),
                    "sha256": sha256_file(shard_path),
                }
            )
    report = {
        "format": "mirage_webdataset_v1",
        "config_sha256": canonical_json_sha256(config.to_dict()),
        "source_manifest_sha256": sha256_file(source / "manifest.jsonl"),
        "samples": len(rows),
        "shards": shards,
        "patterns": {
            split: str(destination / f"{split}-*.tar") for split in ("train", "val", "test")
        },
    }
    (destination / "shard_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def audit_sharded_corpus(config: M3CorpusConfig) -> dict[str, Any]:
    root = Path(config.shards_dir)
    manifest_path = root / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    source_splits: dict[str, set[str]] = {}
    content_hashes: set[str] = set()
    sample_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    audio_samples = 0
    for shard in manifest["shards"]:
        path = Path(shard["path"])
        if not path.is_file():
            errors.append(f"missing shard: {path}")
            continue
        if sha256_file(path) != shard["sha256"]:
            errors.append(f"hash mismatch: {path}")
        grouped: dict[str, set[str]] = {}
        with tarfile.open(path, "r") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                grouped.setdefault(Path(member.name).stem, set()).add(Path(member.name).suffix)
            for sample_id, suffixes in grouped.items():
                if suffixes != {".mp4", ".txt", ".json"}:
                    errors.append(f"incomplete sample {sample_id} in {path}")
                    continue
                metadata_member = archive.getmember(f"{sample_id}.json")
                handle = archive.extractfile(metadata_member)
                if handle is None:
                    errors.append(f"unreadable metadata {sample_id} in {path}")
                    continue
                metadata = json.loads(handle.read())
                if sample_id in sample_ids:
                    errors.append(f"duplicate sample id: {sample_id}")
                sample_ids.add(sample_id)
                split = metadata["split"]
                split_counts[split] += 1
                bucket_counts[metadata["motion_bucket"]] += 1
                source_splits.setdefault(metadata["source_id"], set()).add(split)
                content_hash = metadata["content_sha256"]
                if content_hash in content_hashes:
                    errors.append(f"duplicate content hash: {content_hash}")
                content_hashes.add(content_hash)
                if metadata.get("normalized", {}).get("has_audio"):
                    audio_samples += 1
    leaking = sorted(source for source, splits in source_splits.items() if len(splits) > 1)
    if leaking:
        errors.append(f"source split leakage: {len(leaking)} source(s)")
    if len(sample_ids) != manifest["samples"]:
        errors.append(
            f"sample count mismatch: manifest={manifest['samples']} audited={len(sample_ids)}"
        )
    if config.require_audio and audio_samples != len(sample_ids):
        errors.append(
            f"audio coverage below required 100%: {audio_samples}/{len(sample_ids)} samples"
        )
    if config.target_clips >= 50:
        for split in ("train", "val", "test"):
            if split_counts[split] == 0:
                errors.append(f"required split is empty: {split}")
    report = {
        "format": "mirage_m3_corpus_audit_v1",
        "passed": not errors,
        "samples": len(sample_ids),
        "audio_samples": audio_samples,
        "audio_coverage": audio_samples / max(len(sample_ids), 1),
        "split_counts": dict(split_counts),
        "motion_bucket_counts": dict(bucket_counts),
        "source_split_leaks": leaking,
        "errors": errors,
        "shard_manifest": str(manifest_path),
        "shard_manifest_sha256": sha256_file(manifest_path),
    }
    (root / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
