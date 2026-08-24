import json
from pathlib import Path

import av
import pytest
from PIL import Image

from mirage.reference.composer import (
    LTXRuntimeConfig,
    ReferenceComposer,
    ReferencePipelineConfig,
    ReferenceSpec,
    _grid_shape,
    runtime_report,
)
from mirage.reference.comfy_workflow import build_comfy_workflow


def _image(path: Path, color: str, size: tuple[int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def test_composes_text_free_sheet_static_video_prompt_and_manifest(tmp_path):
    portrait = tmp_path / "person.png"
    landscape = tmp_path / "place.png"
    _image(portrait, "red", (100, 200))
    _image(landscape, "blue", (300, 100))
    config = ReferencePipelineConfig(
        references=(
            ReferenceSpec(str(portrait), "hero", "A hero in a red coat", "character"),
            ReferenceSpec(str(landscape), "station", "A glass railway station", "location"),
        ),
        action_prompt="The hero walks through the station.",
        width=256,
        height=128,
        num_frames=121,
        frame_rate=24,
        panel_gap=4,
        panel_margin=4,
    )
    result = ReferenceComposer(config).compose(tmp_path / "output")

    assert result.sheet_path.is_file()
    assert result.video_path.is_file()
    assert "Reference sheet:" in result.prompt
    assert "Generated video:" in result.prompt
    assert "hero" in result.prompt and "station" in result.prompt
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["references"]) == 2
    assert len(manifest["sha256"]["sheet"]) == 64
    with av.open(str(result.video_path)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 121


def test_config_rejects_short_reference_video(tmp_path):
    image = tmp_path / "ref.png"
    _image(image, "white", (64, 64))
    config = ReferencePipelineConfig(
        references=(ReferenceSpec(str(image), "hero", "A hero", "character"),),
        action_prompt="The hero waves.",
        num_frames=120,
    )
    with pytest.raises(ValueError, match="at least 121"):
        config.validate()


def test_grid_supports_ten_references():
    assert _grid_shape(10) == (3, 4)


def test_runtime_report_identifies_missing_assets(tmp_path):
    image = tmp_path / "ref.png"
    _image(image, "white", (64, 64))
    config = ReferencePipelineConfig(
        references=(ReferenceSpec(str(image), "hero", "A hero", "character"),),
        action_prompt="The hero waves.",
    )
    report = runtime_report(config)
    assert report["ready"] is False
    assert report["no_cpu_offload"] is True


def test_builds_local_ltx25_comfy_workflow(tmp_path):
    comfy_root = Path(
        "C:/Users/rashm/AppData/Local/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI"
    )
    ingredient_template = comfy_root / (
        "custom_nodes/ComfyUI-LTXVideo/example_workflows/2.3/"
        "LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json"
    )
    ltx25_template = comfy_root / "user/default/workflows/LTX 2.5 T 2 V.json"
    if not ingredient_template.is_file() or not ltx25_template.is_file():
        pytest.skip("local ComfyUI-LTXVideo templates are unavailable")

    image = tmp_path / "ref.png"
    _image(image, "white", (256, 128))
    models = Path("C:/Users/rashm/AppData/Local/Comfy-Desktop/ComfyUI-Shared/models")
    # The desktop installation redirects model folders to ComfyUI-Shared. The files do not need
    # to exist for this graph-structure test; they must remain inside their model categories.
    runtime = LTXRuntimeConfig(
        transformer_path=str(models / "diffusion_models/model.safetensors"),
        text_encoder_path=str(models / "text_encoders/text.safetensors"),
        video_vae_path=str(models / "vae/video.safetensors"),
        audio_vae_path=str(models / "vae/audio.safetensors"),
        reference_lora_path=str(models / "loras/edit_anything.safetensors"),
        reference_module_path=str(tmp_path / "edit_anything.module.safetensors"),
        comfy_root=str(comfy_root),
        models_root=str(models),
    )
    config = ReferencePipelineConfig(
        references=(ReferenceSpec(str(image), "hero", "A hero", "character"),),
        action_prompt="The hero waves.",
        runtime=runtime,
    )
    composed = ReferenceComposer(config).compose(tmp_path / "output")
    workflow_path = tmp_path / "workflow.json"
    workflow = build_comfy_workflow(config, composed, workflow_path)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert nodes[3940]["type"] == "DiffusionModelLoaderKJ"
    assert nodes[5023]["type"] == "CLIPLoader"
    assert nodes[6001]["type"] == "VAELoader"
    assert 4922 not in nodes
    assert nodes[5011]["type"] == "MIRAGELTXEditAnythingReference"
    assert nodes[5011]["inputs"][0]["link"] == 13217
    assert next(item for item in nodes[5011]["inputs"] if item["name"] == "reference_sheet")["link"] == 16009
    assert next(item for item in nodes[5011]["inputs"] if item["name"] == "has_reference")["link"] == 16010
    assert nodes[5012]["type"] == "MIRAGELTXOptionalReferenceGuide"
    assert next(item for item in nodes[5012]["inputs"] if item["name"] == "image")["link"] == 16011
    assert nodes[6101]["type"] == "MIRAGEPromptOrSurprise"
    assert nodes[6101]["widgets_values"][0] == ""
    assert nodes[6100]["type"] == "MIRAGEMultiReferenceContactSheet"
    assert [output["name"] for output in nodes[6100]["outputs"]] == [
        "contact_sheet",
        "primary_image",
        "has_reference",
        "reference_manifest",
        "reference_count",
        "zit_denoise",
        "variation_count",
    ]
    assert nodes[6102]["type"] == "MIRAGEGemmaDirector"
    assert nodes[6110]["type"] == "MIRAGELTXFrameCount"
    assert nodes[6103]["type"] == "MIRAGEClearStartFrame"
    assert nodes[7044]["type"] == "KSampler"
    assert nodes[7044]["widgets_values"][-1] == 0.30
    assert nodes[7046]["widgets_values"][0] == "z_image_turbo_int8_convrot.safetensors"
    assert nodes[7048]["widgets_values"][-1] == 1.0
    assert nodes[2483]["inputs"][-1]["name"] == "text"
    assert nodes[2612]["inputs"][-1]["name"] == "text"
    assert 2004 not in nodes
    assert 3159 not in nodes
    assert any(link[:5] == [13217, 3940, 0, 5011, 0] for link in workflow["links"])
    assert any(link[:5] == [13402, 3059, 0, 5012, 3] for link in workflow["links"])
    assert any(link[:5] == [16002, 6102, 2, 2483, 1] for link in workflow["links"])
    assert any(link[:5] == [16104, 6102, 1, 7045, 1] for link in workflow["links"])
    assert any(link[:5] == [16117, 7043, 0, 6103, 4] for link in workflow["links"])
    assert any(link[:5] == [16122, 6100, 5, 7044, 9] for link in workflow["links"])
    assert any(link[:5] == [16123, 6100, 6, 7066, 1] for link in workflow["links"])
    assert any(link[:5] == [16011, 6103, 0, 5012, 4] for link in workflow["links"])
    assert any(link[:5] == [16200, 6102, 6, 6110, 0] for link in workflow["links"])
    assert any(link[:5] == [16202, 6110, 0, 3059, 2] for link in workflow["links"])
    assert any(link[:5] == [16203, 6110, 0, 3980, 1] for link in workflow["links"])
    assert 5072 not in nodes
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert "reroutes" not in workflow["extra"]
    assert "linkExtensions" not in workflow["extra"]
    assert workflow_path.is_file()
