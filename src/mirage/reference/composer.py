from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import av
from PIL import Image, ImageOps


_KINDS = {"character", "prop", "location", "style", "other"}


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    path: str
    label: str
    description: str
    kind: str = "other"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceSpec":
        item = cls(
            path=str(value["path"]),
            label=str(value["label"]).strip(),
            description=str(value["description"]).strip(),
            kind=str(value.get("kind", "other")).lower().strip(),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported reference kind {self.kind!r}; expected {sorted(_KINDS)}")
        if not self.label:
            raise ValueError("reference label cannot be empty")
        if not self.description:
            raise ValueError(f"reference {self.label!r} needs a visual description")
        path = Path(self.path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"reference image not found: {path}")


@dataclass(frozen=True, slots=True)
class LTXRuntimeConfig:
    transformer_path: str = ""
    text_encoder_path: str = ""
    video_vae_path: str = ""
    audio_vae_path: str = ""
    duration_head_path: str = ""
    spatial_upsampler_path: str = ""
    reference_lora_path: str = ""
    reference_module_path: str = ""
    ltx_repo_path: str = ".vendor/LTX-2"
    comfy_root: str = (
        "C:/Users/rashm/AppData/Local/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI"
    )
    models_root: str = "C:/Users/rashm/AppData/Local/Comfy-Desktop/ComfyUI-Shared/models"
    input_root: str = "C:/Users/rashm/AppData/Local/Comfy-Desktop/ComfyUI-Shared/input"
    comfy_input_name: str = "MIRAGE/reference_sheet.png"
    lora_strength: float = 1.4
    conditioning_strength: float = 1.0
    quantization: str = "nvfp4-prequant"
    offload_mode: str = "none"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LTXRuntimeConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: raw for key, raw in value.items() if key in known})

    def paths(self) -> dict[str, str]:
        return {
            "transformer": self.transformer_path,
            "text_encoder": self.text_encoder_path,
            "video_vae": self.video_vae_path,
            "audio_vae": self.audio_vae_path,
            "duration_head": self.duration_head_path,
            "spatial_upsampler": self.spatial_upsampler_path,
            "reference_lora": self.reference_lora_path,
            "reference_module": self.reference_module_path,
            "ltx_repo": self.ltx_repo_path,
            "comfy_root": self.comfy_root,
            "models_root": self.models_root,
            "input_root": self.input_root,
        }


@dataclass(frozen=True, slots=True)
class ReferencePipelineConfig:
    references: tuple[ReferenceSpec, ...]
    action_prompt: str
    width: int = 768
    height: int = 448
    num_frames: int = 121
    frame_rate: int = 24
    seed: int = 0
    panel_gap: int = 8
    panel_margin: int = 8
    max_references: int = 12
    output_video_path: str = "artifacts/reference/output.mp4"
    runtime: LTXRuntimeConfig = field(default_factory=LTXRuntimeConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ReferencePipelineConfig":
        config_path = Path(path)
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        refs = tuple(ReferenceSpec.from_dict(item) for item in raw.pop("references"))
        runtime = LTXRuntimeConfig.from_dict(raw.pop("runtime", {}))
        config = cls(references=refs, runtime=runtime, **raw)
        config.validate()
        return config

    def validate(self) -> None:
        if not 1 <= len(self.references) <= self.max_references:
            raise ValueError(
                f"reference count must be between 1 and {self.max_references}, "
                f"got {len(self.references)}"
            )
        if len({item.label.casefold() for item in self.references}) != len(self.references):
            raise ValueError("reference labels must be unique")
        if not self.action_prompt.strip():
            raise ValueError("action_prompt cannot be empty")
        if self.width % 64 or self.height % 64:
            raise ValueError("LTX output width and height must be divisible by 64")
        if self.num_frames < 121:
            raise ValueError("EditAnything reference experiment must contain at least 121 frames")
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")


@dataclass(frozen=True, slots=True)
class ComposedReference:
    sheet_path: Path
    video_path: Path
    manifest_path: Path
    prompt_path: Path
    prompt: str
    manifest: dict[str, Any]


class ReferenceComposer:
    """Build a compact composite reference sheet for the local EditAnything adapter."""

    def __init__(self, config: ReferencePipelineConfig) -> None:
        config.validate()
        self.config = config

    def compose(self, output_dir: str | Path) -> ComposedReference:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        sheet, panels = self._compose_sheet()
        sheet_path = target / "reference_sheet.png"
        video_path = target / "reference_static.mp4"
        prompt_path = target / "prompt.txt"
        manifest_path = target / "reference_manifest.json"
        sheet.save(sheet_path, format="PNG", optimize=True)
        self._encode_static_video(sheet, video_path)
        prompt = self._build_prompt(panels)
        prompt_path.write_text(prompt, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "mode": "ltx-editanything-composite-reference",
            "sheet": str(sheet_path.resolve()),
            "static_video": str(video_path.resolve()),
            "prompt": prompt,
            "width": self.config.width,
            "height": self.config.height,
            "num_frames": self.config.num_frames,
            "frame_rate": self.config.frame_rate,
            "references": panels,
            "sha256": {
                "sheet": _sha256(sheet_path),
                "static_video": _sha256(video_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return ComposedReference(
            sheet_path=sheet_path,
            video_path=video_path,
            manifest_path=manifest_path,
            prompt_path=prompt_path,
            prompt=prompt,
            manifest=manifest,
        )

    def _compose_sheet(self) -> tuple[Image.Image, list[dict[str, Any]]]:
        rows, columns = _grid_shape(len(self.config.references))
        gap = self.config.panel_gap
        margin = self.config.panel_margin
        available_width = self.config.width - 2 * margin - (columns - 1) * gap
        available_height = self.config.height - 2 * margin - (rows - 1) * gap
        cell_width = available_width // columns
        cell_height = available_height // rows
        if min(cell_width, cell_height) < 64:
            raise ValueError("reference panels are too small for the selected sheet resolution")

        sheet = Image.new("RGB", (self.config.width, self.config.height), "black")
        panels: list[dict[str, Any]] = []
        for index, reference in enumerate(self.config.references):
            row, column = divmod(index, columns)
            left = margin + column * (cell_width + gap)
            top = margin + row * (cell_height + gap)
            with Image.open(Path(reference.path).expanduser()) as source:
                normalized = ImageOps.exif_transpose(source).convert("RGB")
                normalized.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
                x = left + (cell_width - normalized.width) // 2
                y = top + (cell_height - normalized.height) // 2
                sheet.paste(normalized, (x, y))
            panels.append(
                {
                    **asdict(reference),
                    "path": str(Path(reference.path).expanduser().resolve()),
                    "position": _position_name(row, column, rows, columns),
                    "cell": {"x": left, "y": top, "width": cell_width, "height": cell_height},
                }
            )
        return sheet, panels

    def _build_prompt(self, panels: list[dict[str, Any]]) -> str:
        descriptions = " ".join(
            f"{panel['position']} ({panel['kind'].title()}, {panel['label']}): "
            f"{panel['description']}"
            for panel in panels
        )
        return f"Reference sheet: {descriptions}\n\nGenerated video: {self.config.action_prompt.strip()}"

    def _encode_static_video(self, sheet: Image.Image, path: Path) -> None:
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=self.config.frame_rate)
            stream.width = self.config.width
            stream.height = self.config.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium", "tune": "stillimage"}
            frame = av.VideoFrame.from_image(sheet)
            for _ in range(self.config.num_frames):
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


def runtime_report(config: ReferencePipelineConfig) -> dict[str, Any]:
    assets = {}
    for name, raw_path in config.runtime.paths().items():
        path = Path(raw_path).expanduser() if raw_path else None
        assets[name] = {
            "path": str(path.resolve()) if path else "",
            "exists": bool(path and path.exists()),
            "bytes": path.stat().st_size if path and path.is_file() else None,
        }
    required = (
        "transformer",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "duration_head",
        "spatial_upsampler",
        "reference_lora",
        "reference_module",
        "comfy_root",
        "models_root",
        "input_root",
    )
    return {
        "ready": all(assets[name]["exists"] for name in required),
        "offload_mode": config.runtime.offload_mode,
        "no_cpu_offload": config.runtime.offload_mode.lower() == "none",
        "quantization": config.runtime.quantization,
        "assets": assets,
    }


def _grid_shape(count: int) -> tuple[int, int]:
    columns = min(4, math.ceil(math.sqrt(count * 16 / 9)))
    rows = math.ceil(count / columns)
    while columns > 1 and (columns - 1) * rows >= count:
        columns -= 1
    return rows, columns


def _position_name(row: int, column: int, rows: int, columns: int) -> str:
    row_name = "Row" if rows == 1 else ("Top row" if row == 0 else "Bottom row" if row == rows - 1 else f"Row {row + 1}")
    if columns == 1:
        return row_name
    if columns == 2:
        column_name = "left" if column == 0 else "right"
    elif columns == 3:
        column_name = ("left", "middle", "right")[column]
    else:
        column_name = f"column {column + 1}"
    return f"{row_name} {column_name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
