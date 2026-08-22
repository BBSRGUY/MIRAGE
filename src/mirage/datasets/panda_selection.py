from __future__ import annotations

import ast
import csv
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..m3_data_config import M3CorpusConfig
from ..training.provenance import canonical_json_sha256, sha256_file


def parse_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    seconds = float(parts[-1])
    minutes = float(parts[-2])
    hours = float(parts[-3]) if len(parts) > 2 else 0.0
    return hours * 3600 + minutes * 60 + seconds


def _literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return value


def _at(value: Any, index: int) -> Any:
    parsed = _literal(value)
    if isinstance(parsed, (list, tuple)):
        if not parsed:
            return None
        return parsed[min(index, len(parsed) - 1)]
    return parsed


def _source_id(row: dict[str, Any]) -> str:
    return str(row.get("videoID") or row.get("video_id") or row.get("url") or "")


def _bucket(caption: str, config: M3CorpusConfig) -> str:
    lowered = caption.lower()
    for bucket in config.motion_buckets:
        if any(keyword in lowered for keyword in bucket.keywords):
            return bucket.name
    digest = int(hashlib.blake2b(caption.encode("utf-8"), digest_size=4).hexdigest(), 16)
    cumulative = 0.0
    point = (digest % 1_000_000) / 1_000_000
    for bucket in config.motion_buckets:
        cumulative += bucket.fraction
        if point <= cumulative:
            return bucket.name
    return config.motion_buckets[-1].name


def _split(source_id: str, config: M3CorpusConfig) -> str:
    point = int(
        hashlib.blake2b(f"{config.seed}:{source_id}".encode(), digest_size=8).hexdigest(), 16
    ) % 1_000_000 / 1_000_000
    if point < config.test_fraction:
        return "test"
    if point < config.test_fraction + config.validation_fraction:
        return "val"
    return "train"


@dataclass(order=True)
class _Candidate:
    priority: int
    payload: dict[str, Any] = field(compare=False)


def _shot_count(value: Any, index: int, clip_count: int) -> int:
    parsed = _literal(value)
    if not isinstance(parsed, (list, tuple)) or not parsed:
        return 1
    # Panda annotations may be either one interval list per clip or a nested
    # list of intervals per clip. Avoid treating [start, end] as two shots.
    selected = parsed[index] if len(parsed) == clip_count else parsed
    if (
        isinstance(selected, (list, tuple))
        and len(selected) == 2
        and all(isinstance(item, (int, float)) for item in selected)
    ):
        return 1
    if isinstance(selected, (list, tuple)):
        return len(selected)
    return 1


def _bucket_quotas(config: M3CorpusConfig) -> dict[str, int]:
    raw = [config.target_clips * bucket.fraction for bucket in config.motion_buckets]
    quotas = [int(value) for value in raw]
    remainder = config.target_clips - sum(quotas)
    order = sorted(range(len(raw)), key=lambda index: (raw[index] % 1, -index), reverse=True)
    for index in order[:remainder]:
        quotas[index] += 1
    return {bucket.name: quota for bucket, quota in zip(config.motion_buckets, quotas)}


def _clip_rows(row: dict[str, Any], config: M3CorpusConfig):
    timestamps = _literal(row.get("timestamp"))
    if not isinstance(timestamps, (list, tuple)):
        return
    source_id = _source_id(row)
    if not source_id:
        return
    for index, interval in enumerate(timestamps):
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            continue
        try:
            start, end = parse_time(interval[0]), parse_time(interval[1])
        except (TypeError, ValueError):
            yield None, "timestamp"
            continue
        duration = end - start
        caption = str(_at(row.get("caption"), index) or "").strip()
        score_raw = _at(row.get("matching_score"), index)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        desirable = str(_at(row.get("desirable_filtering"), index) or "").lower()
        shot_count = _shot_count(row.get("shot_boundary_detection"), index, len(timestamps))
        reason = None
        if not config.duration_min_s <= duration <= config.duration_max_s:
            reason = "duration"
        elif score < config.matching_score_min:
            reason = "matching_score"
        elif len(caption) < 12:
            reason = "caption"
        elif config.require_desirable and desirable != "desirable":
            reason = "desirability"
        elif shot_count > config.max_shot_segments:
            reason = "shot_count"
        if reason:
            yield None, reason
            continue
        stable = f"{source_id}:{start:.3f}:{end:.3f}"
        sample_id = hashlib.blake2b(stable.encode(), digest_size=10).hexdigest()
        payload = {
            "sample_id": sample_id,
            "source": config.source,
            "source_id": source_id,
            "url": row.get("url") or f"https://www.youtube.com/watch?v={source_id}",
            "start": start,
            "end": end,
            "duration": duration,
            "caption": caption,
            "matching_score": score,
            "desirable_filtering": desirable or None,
            "shot_count": shot_count,
            "motion_bucket": _bucket(caption, config),
            "split": _split(source_id, config),
        }
        priority = int(
            hashlib.blake2b(f"{config.seed}:{stable}".encode(), digest_size=8).hexdigest(), 16
        )
        yield (priority, payload), None


def select_panda_metadata(config: M3CorpusConfig) -> dict[str, Any]:
    source = Path(config.source_metadata)
    output = Path(config.work_dir) / "selection"
    output.mkdir(parents=True, exist_ok=True)
    quotas = _bucket_quotas(config)
    heaps: dict[str, list[_Candidate]] = defaultdict(list)
    rejects: Counter[str] = Counter()
    seen_sources: Counter[str] = Counter()
    rows_seen = clips_seen = 0
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_seen += 1
            for candidate, reason in _clip_rows(row, config):
                clips_seen += 1
                if candidate is None:
                    rejects[reason or "unknown"] += 1
                    continue
                priority, payload = candidate
                source_id = payload["source_id"]
                if seen_sources[source_id] >= config.max_clips_per_source:
                    rejects["source_cap"] += 1
                    continue
                seen_sources[source_id] += 1
                heap = heaps[payload["motion_bucket"]]
                if quotas[payload["motion_bucket"]] == 0:
                    rejects["bucket_quota_zero"] += 1
                    continue
                item = _Candidate(-priority, payload)
                if len(heap) < quotas[payload["motion_bucket"]]:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    selected = sorted(
        (item.payload for heap in heaps.values() for item in heap),
        key=lambda row: (row["split"], row["motion_bucket"], row["sample_id"]),
    )
    selection_path = output / "selected.jsonl"
    request_path = output / "panda_download.csv"
    selection_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8"
    )
    with request_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "url",
                "caption",
                "timestamp",
                "matching_score",
                "desirable_filtering",
                "shot_boundary_detection",
                "mirage_id",
                "motion_bucket",
                "split",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "url": row["url"],
                    "caption": repr([row["caption"]]),
                    "timestamp": repr([[row["start"], row["end"]]]),
                    "matching_score": repr([row["matching_score"]]),
                    "desirable_filtering": repr([row["desirable_filtering"]]),
                    "shot_boundary_detection": repr([[0, row["duration"]]]),
                    "mirage_id": row["sample_id"],
                    "motion_bucket": row["motion_bucket"],
                    "split": row["split"],
                }
            )
    report = {
        "format": "mirage_m3_selection_v1",
        "config_sha256": canonical_json_sha256(config.to_dict()),
        "source_metadata": str(source),
        "source_metadata_sha256": sha256_file(source),
        "rows_seen": rows_seen,
        "clips_seen": clips_seen,
        "selected": len(selected),
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "motion_bucket_counts": dict(Counter(row["motion_bucket"] for row in selected)),
        "reject_counts": dict(rejects),
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "download_request": str(request_path),
        "download_request_sha256": sha256_file(request_path),
        "video2dataset_command": [
            "video2dataset",
            f"--url_list={request_path}",
            "--url_col=url",
            "--caption_col=caption",
            "--clip_col=timestamp",
            f"--output_folder={config.downloaded_dir}",
            "--save_additional_columns=[matching_score,desirable_filtering,shot_boundary_detection,mirage_id,motion_bucket,split]",
            "--config=video2dataset/video2dataset/configs/panda70m.yaml",
        ],
        "license": "Panda-70M research use; source-video licenses remain individually binding",
    }
    (output / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
