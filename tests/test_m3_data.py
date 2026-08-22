import csv
import json
import subprocess

from mirage.config import MirageConfig
from mirage.datasets.corpus_builder import (
    audit_sharded_corpus,
    normalize_downloaded_corpus,
    shard_normalized_corpus,
)
from mirage.datasets.m3_stream import StreamingAVDataset
from mirage.datasets.panda_selection import select_panda_metadata
from mirage.m3_config import M3Config, M3DataConfig
from mirage.m3_data_config import M3CorpusConfig
from mirage.training.provenance import build_run_provenance


def test_m3_corpus_build_audit_load_and_provenance(tmp_path):
    source = tmp_path / "panda.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "videoID",
                "url",
                "timestamp",
                "caption",
                "matching_score",
                "desirable_filtering",
                "shot_boundary_detection",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "videoID": "source-one",
                "url": "https://example.invalid/source-one",
                "timestamp": repr([[0, 2.2]]),
                "caption": repr(["A person walks across a colorful moving scene."]),
                "matching_score": repr([0.9]),
                "desirable_filtering": repr(["desirable"]),
                "shot_boundary_detection": repr([[0, 2.2]]),
            }
        )
    work = tmp_path / "corpus"
    config = M3CorpusConfig(
        source_metadata=str(source),
        work_dir=str(work),
        downloaded_dir=str(work / "downloaded"),
        normalized_dir=str(work / "normalized"),
        shards_dir=str(work / "shards"),
        target_clips=1,
        validation_fraction=0.0,
        test_fraction=0.0,
        motion_difference_min=0.0,
        motion_difference_max=1.0,
        normalized_size=64,
        shard_size=1,
    )
    selection = select_panda_metadata(config)
    assert selection["selected"] == 1
    row = json.loads((work / "selection" / "selected.jsonl").read_text().strip())

    downloaded = work / "downloaded"
    downloaded.mkdir()
    video = downloaded / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x320:rate=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            "2.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
        timeout=30,
    )
    video.with_suffix(".json").write_text(
        json.dumps({"mirage_id": row["sample_id"]}), encoding="utf-8"
    )
    normalized = normalize_downloaded_corpus(config)
    assert normalized["accepted"] == 1
    resumed = normalize_downloaded_corpus(config)
    assert resumed["resumed_samples"] == 1
    assert resumed["accepted"] == 1
    sharded = shard_normalized_corpus(config)
    assert sharded["samples"] == 1
    audit = audit_sharded_corpus(config)
    assert audit["passed"] is True
    assert audit["audio_coverage"] == 1.0

    manifest = f"webdataset://{work / 'shards' / '{split}-*.tar'}"
    train_config = M3Config(
        model=MirageConfig(
            frames=4,
            height=32,
            width=32,
            projection_backend="independent",
            cache_threshold=0.0,
            vram_budget_gb=20.0,
        ),
        data=M3DataConfig(manifest=manifest, teacher_features=None),
    )
    sample = next(iter(StreamingAVDataset(train_config, "train")))
    assert sample["video"].shape == (4, 3, 32, 32)
    assert sample["audio"].shape == (4 * 320,)
    provenance = build_run_provenance(train_config)
    assert provenance["dataset"]["webdataset"] is True
    assert provenance["dataset"]["files"] == 1
