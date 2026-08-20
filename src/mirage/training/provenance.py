from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from ..m3_config import M3Config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_file_set(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if not root.exists():
        return {"root": str(root), "exists": False, "sha256": None, "files": 0}
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return {
        "root": str(root),
        "exists": True,
        "sha256": digest.hexdigest(),
        "files": len(files),
    }


def hash_model_state(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def code_provenance(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    commit = _git_value("rev-parse", "HEAD")
    status = _git_value("status", "--porcelain", "--untracked-files=all")
    files = [root / "pyproject.toml"]
    files.extend(sorted((root / "src").rglob("*.py")))
    digest = hashlib.sha256()
    count = 0
    for path in files:
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
        count += 1
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "code_snapshot_sha256": digest.hexdigest(),
        "code_files": count,
    }


def build_run_provenance(config: M3Config, root: str | Path = ".") -> dict[str, Any]:
    manifest = config.data.manifest
    dataset = (
        {"manifest": manifest, "synthetic": True, "sha256": canonical_json_sha256(manifest)}
        if manifest.startswith("synthetic://")
        else {
            "manifest": manifest,
            "synthetic": False,
            "sha256": sha256_file(manifest),
        }
    )
    teacher = (
        hash_file_set(config.data.teacher_features)
        if config.data.teacher_features
        else {"root": None, "exists": False, "sha256": None, "files": 0}
    )
    canonical_config = config.to_dict()
    # Resume location is operational state, not part of the experiment definition.
    canonical_config["training"]["resume"] = None
    return {
        "format": "mirage_provenance_v1",
        "config_sha256": canonical_json_sha256(canonical_config),
        "dataset": dataset,
        "teacher_features": teacher,
        "code": code_provenance(root),
    }


def comparable_run_inputs(provenance: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match to claim a bitwise-continuous resumed experiment."""
    return {
        "config_sha256": provenance["config_sha256"],
        "dataset_sha256": provenance["dataset"]["sha256"],
        "teacher_features_sha256": provenance["teacher_features"]["sha256"],
        "code_snapshot_sha256": provenance["code"]["code_snapshot_sha256"],
    }
