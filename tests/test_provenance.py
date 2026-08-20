import json

import torch

from mirage.config import MirageConfig
from mirage.m3_config import M3Config, M3DataConfig, M3TrainingConfig
from mirage.training.provenance import (
    build_run_provenance,
    canonical_json_sha256,
    hash_file_set,
    hash_model_state,
)


def test_canonical_config_hash_ignores_key_order():
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256({"b": 2, "a": 1})


def test_file_set_and_model_hash_change_with_content(tmp_path):
    target = tmp_path / "feature.safetensors"
    target.write_bytes(b"first")
    first = hash_file_set(tmp_path)["sha256"]
    target.write_bytes(b"second")
    assert hash_file_set(tmp_path)["sha256"] != first
    model = torch.nn.Linear(4, 4)
    before = hash_model_state(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert hash_model_state(model) != before


def test_run_provenance_hashes_manifest_and_teacher_set(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"sample_id": "one", "split": "train"}), encoding="utf-8")
    features = tmp_path / "teacher"
    features.mkdir()
    (features / "manifest.json").write_text("{}", encoding="utf-8")
    config = M3Config(
        model=MirageConfig(
            projection_backend="independent", cache_threshold=0.0, vram_budget_gb=20.0
        ),
        data=M3DataConfig(manifest=str(manifest), teacher_features=str(features)),
        training=M3TrainingConfig(output_dir=str(tmp_path / "output")),
    )
    provenance = build_run_provenance(config)
    assert provenance["dataset"]["sha256"]
    assert provenance["teacher_features"]["files"] == 1
    assert provenance["code"]["code_snapshot_sha256"]
