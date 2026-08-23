from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from .composer import ComposedReference, ReferencePipelineConfig


_INGREDIENTS_TEMPLATE = (
    "custom_nodes/ComfyUI-LTXVideo/example_workflows/2.3/"
    "LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json"
)
_LTX25_LOADER_TEMPLATE = "user/default/workflows/LTX 2.5 T 2 V.json"


def build_comfy_workflow(
    config: ReferencePipelineConfig,
    composed: ComposedReference,
    output_path: str | Path,
) -> dict[str, Any]:
    """Adapt the local IC-LoRA graph to LTX-2.5 plus the MIRAGE EditAnything node."""
    comfy_root = Path(config.runtime.comfy_root).expanduser().resolve()
    models_root = Path(config.runtime.models_root).expanduser().resolve()
    ingredient_path = comfy_root / _INGREDIENTS_TEMPLATE
    ltx25_path = comfy_root / _LTX25_LOADER_TEMPLATE
    if not ingredient_path.is_file():
        raise FileNotFoundError(f"ComfyUI-LTXVideo Ingredients workflow not found: {ingredient_path}")
    if not ltx25_path.is_file():
        raise FileNotFoundError(f"local LTX-2.5 workflow not found: {ltx25_path}")

    workflow = json.loads(ingredient_path.read_text(encoding="utf-8"))
    ltx25 = json.loads(ltx25_path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    source_nodes = {node["id"]: node for node in ltx25["nodes"]}
    _expect(nodes, 3940, "CheckpointLoaderSimple")
    _expect(nodes, 4922, "LoraLoaderModelOnly")
    _expect(nodes, 5011, "LTXICLoRALoaderModelOnly")
    _expect(nodes, 5012, "LTXAddVideoICLoRAGuide")
    _expect(nodes, 5023, "LTXAVTextEncoderLoader")
    _expect(source_nodes, 455, "DiffusionModelLoaderKJ")
    _expect(source_nodes, 442, "CLIPLoader")
    _expect(source_nodes, 440, "VAELoader")
    _expect(source_nodes, 441, "VAELoader")

    model_node = _loader_clone(source_nodes[455], nodes[3940], 3940)
    model_node["widgets_values"][0] = _model_name(
        config.runtime.transformer_path, models_root / "diffusion_models"
    )
    model_node["outputs"][0]["links"] = [13217]

    clip_node = _loader_clone(source_nodes[442], nodes[5023], 5023)
    clip_node["widgets_values"][0] = _model_name(
        config.runtime.text_encoder_path, models_root / "text_encoders"
    )
    clip_node["outputs"][0]["links"] = [13459, 13460]

    video_vae_node = copy.deepcopy(source_nodes[440])
    video_vae_node.update(
        {
            "id": 6001,
            "pos": [nodes[3940]["pos"][0], nodes[3940]["pos"][1] + 130],
            "order": nodes[3940].get("order", 0) + 1,
        }
    )
    video_vae_node["widgets_values"][0] = _model_name(
        config.runtime.video_vae_path, models_root / "vae"
    )
    video_vae_node["outputs"][0]["links"] = [13279, 13405, 13543]

    audio_node = _loader_clone(source_nodes[441], nodes[4010], 4010)
    audio_node["widgets_values"][0] = _model_name(
        config.runtime.audio_vae_path, models_root / "vae"
    )
    audio_node["outputs"][0]["links"] = nodes[4010]["outputs"][0]["links"]

    workflow["nodes"] = [
        node for node in workflow["nodes"] if node["id"] not in {3940, 4922, 5023, 4010}
    ]
    workflow["nodes"].extend([model_node, clip_node, video_vae_node, audio_node])
    nodes = {node["id"]: node for node in workflow["nodes"]}

    nodes[2004]["widgets_values"][0] = config.runtime.comfy_input_name
    nodes[2483]["widgets_values"] = composed.prompt
    reference_lora = Path(config.runtime.reference_lora_path).expanduser().resolve()
    lora_roots = [models_root / "loras", reference_lora.parent]
    lora_name = next(
        (
            reference_lora.relative_to(root.resolve()).as_posix()
            for root in lora_roots
            if reference_lora.is_relative_to(root.resolve())
        ),
        reference_lora.name,
    )
    nodes[5011].update(
        {
            "type": "MIRAGELTXEditAnythingReference",
            "size": [560, 300],
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 13217},
                {"name": "vae", "type": "VAE", "link": 16000},
                {"name": "reference_sheet", "type": "IMAGE", "link": 16001},
            ],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [13401]},
                {"name": "reference_latent", "type": "LATENT", "links": []},
            ],
            "properties": {
                "Node name for S&R": "MIRAGELTXEditAnythingReference",
                "mirage_adapter": "wan2gp-editanything-local-port",
            },
            "widgets_values": [
                lora_name,
                str(Path(config.runtime.reference_module_path).expanduser().resolve()),
                config.runtime.lora_strength,
                0.01,
                0.25,
                2.0,
                12,
                35,
            ],
        }
    )
    nodes[5072]["widgets_values"] = [config.num_frames, "fixed"]
    nodes[5069]["widgets_values"] = ["scale shorter dimension", config.height, "lanczos"]
    nodes[5098]["widgets_values"] = config.frame_rate
    nodes[4832]["widgets_values"][0] = config.seed

    patched_links = []
    for link in workflow["links"]:
        if link[0] in {13400, 13406}:
            continue
        if link[0] == 13217:
            link = [13217, 3940, 0, 5011, 0, "MODEL"]
        elif link[1] == 3940 and link[2] == 2:
            link = [link[0], 6001, 0, link[3], link[4], link[5]]
        patched_links.append(link)
    workflow["links"] = patched_links
    workflow["links"].extend(
        [
            [16000, 6001, 0, 5011, 1, "VAE"],
            [16001, 5093, 0, 5011, 2, "IMAGE"],
        ]
    )
    nodes[6001]["outputs"][0]["links"].append(16000)
    nodes[5093]["outputs"][0]["links"].append(16001)
    guide_downscale = next(item for item in nodes[5012]["inputs"] if item["name"] == "latent_downscale_factor")
    guide_downscale["link"] = None
    # Some published workflow templates retain UI-only links to nodes that are no
    # longer serialized. Remove those dangling links so API validation is strict.
    node_ids = {node["id"] for node in workflow["nodes"]}
    workflow["links"] = [
        link for link in workflow["links"] if link[1] in node_ids and link[3] in node_ids
    ]
    link_ids = {link[0] for link in workflow["links"]}
    for node in workflow["nodes"]:
        for output in node.get("outputs", []):
            if output.get("links") is not None:
                output["links"] = [link for link in output["links"] if link in link_ids]
    # Reroutes are optional canvas decoration. The upstream template contains
    # reroutes for links replaced above; stale reroute parents can make the
    # frontend abort canvas restoration and display an empty graph.
    workflow.setdefault("extra", {}).pop("reroutes", None)
    workflow["extra"].pop("linkExtensions", None)
    workflow["last_node_id"] = max(workflow.get("last_node_id", 0), 6001)
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    workflow.setdefault("extra", {})["mirage_reference_manifest"] = str(
        composed.manifest_path.resolve()
    )
    workflow["extra"]["mirage_backend"] = "ltx-2.5-nvfp4-editanything-local-port"

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    return workflow


def deploy_comfy_assets(
    config: ReferencePipelineConfig,
    composed: ComposedReference,
    workflow_path: str | Path,
) -> dict[str, str]:
    comfy_root = Path(config.runtime.comfy_root).expanduser().resolve()
    input_root = Path(config.runtime.input_root).expanduser().resolve()
    input_target = input_root / Path(config.runtime.comfy_input_name)
    workflow_target = comfy_root / "user" / "default" / "workflows" / "MIRAGE LTX2.5 Ref2V.json"
    input_target.parent.mkdir(parents=True, exist_ok=True)
    workflow_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(composed.sheet_path, input_target)
    shutil.copy2(workflow_path, workflow_target)
    repo_root = Path(__file__).resolve().parents[3]
    node_source = repo_root / "comfy_nodes" / "mirage_ltx_reference" / "__init__.py"
    node_target = comfy_root / "custom_nodes" / "mirage_ltx_reference" / "__init__.py"
    node_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(node_source, node_target)
    return {
        "input_sheet": str(input_target),
        "workflow": str(workflow_target),
        "custom_node": str(node_target),
    }


def _loader_clone(source: dict[str, Any], target: dict[str, Any], node_id: int) -> dict[str, Any]:
    result = copy.deepcopy(source)
    result.update(
        {
            "id": node_id,
            "pos": target.get("pos", source.get("pos")),
            "size": target.get("size", source.get("size")),
            "order": target.get("order", source.get("order", 0)),
            "mode": target.get("mode", 0),
        }
    )
    return result


def _model_name(raw_path: str, category_root: Path) -> str:
    path = Path(raw_path).expanduser().resolve()
    try:
        return path.relative_to(category_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"model must be inside {category_root}: {path}") from error


def _expect(nodes: dict[int, dict[str, Any]], node_id: int, expected: str) -> None:
    actual = nodes.get(node_id, {}).get("type")
    if actual != expected:
        raise ValueError(f"workflow node {node_id} changed: expected {expected}, found {actual}")
